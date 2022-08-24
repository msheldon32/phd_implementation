import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import sklearn
import math
import re
import os
from datetime import datetime
import random
import sqlite3
import gc
import json
import scipy.integrate as spi
import time
import copy
import csv

import sklearn.cluster

import spatial_decomp_station
import comp
from oslo_data import *
import large_fluid

# Parameters
TIME_POINTS_PER_HOUR = 100
ATOL = 10**(-6)

def generate_uniform(interval):
    lo, hi = interval
    r = hi-lo
    return random.random()*r + lo

def generate_dis_uniform(interval):
    lo, hi = interval
    return random.randint(lo, hi)

def generate_gaussian(moments):
    # moments: [mean, std. dev]
    return np.random.normal(moments[0], moments[1])

def euclidian_distance(a, b):
    sum_squares = 0

    for aval, bval in zip(a,b):
        sum_squares += (aval-bval)**2
    
    return sum_squares ** 0.5

class ExperimentModel:
    def __init__(self, stations, durations, demands, starting_bps, seed):
        self.n_stations = len(stations)

        self.stations  = stations
        self.durations = durations
        self.demands   = demands
        
        self.starting_bps = starting_bps # starting bikes per station
        self.total_bikes = sum(self.starting_bps)

        self.n_cells = 1
        self.station_to_cell = [0 for i in range(self.n_stations)]
        self.cell_to_station = [set([i for i in range(self.n_stations)])]

        self.seed = seed
    
    def generate_cells(self, stations_per_cell):
        self.n_cells = round(self.n_stations/stations_per_cell)

        self.cell_to_station = [set() for i in range(self.n_cells)]

        clusterer = sklearn.cluster.KMeans(n_clusters=self.n_cells, random_state=self.seed)
        clusterer.fit(self.stations)

        self.station_to_cell = [x for x in clusterer.predict(self.stations)]
        
        for i, cell in enumerate(self.station_to_cell):
            self.cell_to_station[cell].add(i)
    
    def get_dt_from_ratio(self, ratio):
        avg_rate = [0,0]

        for i in range(self.n_stations):
            for j in range(self.n_stations):
                avg_rate[0] += self.demands[i][j]
                avg_rate[1] += 1
        return ratio/(avg_rate[0]/avg_rate[1])
    
    def get_delay_idx(self, srt_stn, end_stn):
        return (srt_stn*self.n_stations) + end_stn
    
    def get_delay_start(self, dly_idx):
        return dly_idx // self.n_stations


class ExperimentConfig:
    def __init__(self):
        # meta
        self.repetitions_per_point = 300

        # parameters
        self.ode_methods             = ["BDF", "RK45"]
        self.stations_per_cell       = [5, 10, 15, 20]
        self.delta_t_ratio           = [1, 0.5, 0.1] # setting delta T based on (x/[average rate])

        # random configuration
        self.n_station_range         = [25, 50]
        self.x_location_range        = [0, 1]
        self.y_location_range        = [0, 1]
        self.station_demand_range    = [0, 0.5]
        self.noise_moments_distance  = [0, 0.2]
        self.bps_range               = [0, 15]

        # constants
        self.n_iterations = 25
        self.time_end     = 24
        self.min_duration = 0.01
        self.steps_per_dt = 100

        self.seed = 3#1996

    
    def generate_stations(self):
        n_stations = round(generate_uniform(self.n_station_range))

        station_list = []

        for station_no in range(n_stations):
            x_loc = generate_uniform(self.x_location_range)
            y_loc = generate_uniform(self.y_location_range)
            station_list.append([x_loc, y_loc])
        
        return station_list
    
    def generate_durations(self, stations):
        n_stations = len(stations)
        durations = [[0 for i in range(n_stations)] for j in range(n_stations)]

        for i, src_stn in enumerate(stations):
            for j, dst_stn in enumerate(stations):
                distance = euclidian_distance(src_stn, dst_stn)
                eps = generate_gaussian(self.noise_moments_distance)
                durations[i][j] = max(distance + eps, self.min_duration)
        return durations

    def generate_demands(self, stations):
        n_stations = len(stations)
        demands = [[0 for i in range(n_stations)] for j in range(n_stations)]

        for i, src_stn in enumerate(stations):
            for j, dst_stn in enumerate(stations):
                demands[i][j] = generate_uniform(self.station_demand_range)
        
        return demands

    def generate_bps(self, stations):
        n_stations = len(stations)

        bps = [generate_dis_uniform(self.bps_range) for i in range(n_stations)]
        return bps

    def generate_model(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed += 1

        stations = self.generate_stations()
        durations = self.generate_durations(stations)
        demands = self.generate_demands(stations)
        starting_bps = self.generate_bps(stations)

        return ExperimentModel(stations, durations, demands, starting_bps, self.seed)
    
def refit_times(base_t, base_y, alt_t):
    t_map = [min(np.searchsorted(base_t, t), len(base_t)-1) for t in alt_t]
    return base_y[:, t_map]

def get_accuracy_score(base_t, base_y, alt_t, alt_y):
    diff = abs(refit_times(base_t, base_y, alt_t) - alt_y)
    total_val = base_y[:,0].sum()
    return (diff/total_val).max()

class Experiment:
    def __init__(self, configuration):
        self.configuration = configuration

        self.output_folder = ""
    
    def run_full(self, model, ode_method):
        print("Running Full Model")
        tic = time.perf_counter()

        comp_model = comp.CompModel(model.durations, model.demands)

        n_time_points = self.configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*self.configuration.time_end)/n_time_points for i in range(n_time_points+1)]

        starting_vector = [0 for i in range(comp_model.n_stations**2)] + model.starting_bps
        
        x_t = spi.solve_ivp(comp_model.dxdt, [0, self.configuration.time_end], starting_vector, 
                                t_eval=time_points, 
                                method=ode_method, atol=ATOL)
        toc = time.perf_counter()
        print(f"Full Model finished, time: {toc-tic}.")
        return [[x for x in x_t.t], x_t.y, toc-tic]
    
    def run_iteration(self, model, ode_method):
        print("Running Trajectory-Based Iteration")

        n_entries = self.n_stations**2 + self.n_stations

        tic = time.perf_counter()

        x_res = []

        starting_vector = [
            [0 for i in range(len(model.cell_to_station[cell_idx])*model.n_stations)] +
            [x for i, x in enumerate(model.starting_bps) if i in model.cell_to_station[cell_idx]]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_station.TrajCell(cell_idx, model.station_to_cell, model.cell_to_station, 
                                    sorted(list(model.cell_to_station[cell_idx])), model.durations, model.demands)
                            for cell_idx in range(model.n_cells)]

        trajectories = [[(lambda t: 0) for j in range(model.n_stations)] for i in range(model.n_stations)]

        n_time_points = self.configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*self.configuration.time_end)/n_time_points for i in range(n_time_points+1)]

        station_vals = []

        for iter_no in range(self.configuration.n_iterations):
            for tc in traj_cells:
                tc.set_trajectories(trajectories)

            new_trajectories = copy.deepcopy(trajectories)

            total_bikes = 0

            x_iter = np.array([[0 for i in range(n_time_points+1)] for i in range(n_entries)])

            for cell_idx in range(model.n_cells):
                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt, [0, self.configuration.time_end], starting_vector[cell_idx], 
                                        t_eval=time_points, 
                                        method=ode_method, atol=ATOL)
                x_iter[traj_cells[cell_idx].get_idx()] = x_t.y
                

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    sy_idx = traj_cells[cell_idx].get_station_idx(i)
                    
                    for dst_stn in range(model.n_stations):
                        y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        new_trajectories[src_stn][dst_stn] = spatial_decomp_station.get_traj_fn(x_t.t, x_t.y[y_idx, :])
                total_bikes += sum(x_t.y[:,-1])
            
            x_res.append(x_iter)

            trajectories = new_trajectories
        

        toc = time.perf_counter()
        print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

        return [time_points, x_res, toc-tic]
    
    def run_discrete(self, model, ode_method, step_size):
        print("Running Discrete-Step Submodeling")

        tic = time.perf_counter()

        x_arr = np.array([])

        starting_vector = [
            [0 for i in range(len(model.cell_to_station[cell_idx])*model.n_stations)] +
            [x for i, x in enumerate(model.starting_bps) if i in model.cell_to_station[cell_idx]]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_station.TrajCell(cell_idx, model.station_to_cell, model.cell_to_station, 
                                    sorted(list(model.cell_to_station[cell_idx])), model.durations, model.demands)
                            for cell_idx in range(model.n_cells)]

        trajectories = [[(lambda t: 0) for j in range(model.n_stations)] for i in range(model.n_stations)]

        station_vals = [[] for i in range(model.n_stations)]

        time_points = []
        
        current_vector = copy.deepcopy(starting_vector)

        t = 0

        
        while t < self.configuration.time_end:
            sub_time_points = [t+(i*(step_size/self.configuration.steps_per_dt)) for i in range(self.configuration.steps_per_dt)]
            if len(sub_time_points) == 0:
                sub_time_points.append(t)
            time_points += sub_time_points
            
            new_trajectories = copy.deepcopy(trajectories)
            new_vector = copy.deepcopy(current_vector)


            x_iter = np.array([[0 for x in range(len(sub_time_points))] for i in range(model.n_stations)])

            for cell_idx in range(model.n_cells):
                traj_cells[cell_idx].set_trajectories(trajectories)

                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt, [t, t+step_size], current_vector[cell_idx], 
                                        t_eval = sub_time_points,
                                        method=ode_method, atol=ATOL)
                                        
                if x_iter.size == 0:
                    x_iter = x_t.y
                else:
                    x_iter = np.concatenate([x_iter, x_t.y[:,:x_iter.shape[1]]], axis=0)

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    sy_idx = traj_cells[cell_idx].get_station_idx(i)
                    
                    station_vals[src_stn] += [comp.get_traj_fn(x_t.t, x_t.y[sy_idx,:])(t) for t in sub_time_points]
                    
                    new_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

                    for dst_stn in range(model.n_stations):
                        y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        last_val = float(x_t.y[y_idx, -1])
                        new_vector[cell_idx][y_idx] = last_val
                        delay_rate = 1/model.durations[src_stn][dst_stn]
                        traj = spatial_decomp_station.get_traj_fn_lval(last_val)
                        new_trajectories[src_stn][dst_stn] = traj

            if x_arr.size == 0:
                x_arr = x_iter
            else:
                x_arr = np.concatenate([x_arr, x_iter], axis=1)

            trajectories = new_trajectories
            current_vector = new_vector

            t += step_size
        toc = time.perf_counter()
        print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
        return [time_points, x_arr, toc-tic]

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(self.output_folder + "output.csv", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["solution_method", "spatial_simplification", "ode_method", "stations_per_cell", 
                             "iteration", "delta_t", "station_no", "delta"])
        try:
            for repetition in range(self.configuration.repetitions_per_point):
                print(f"Repetition: {repetition}")
                model = self.configuration.generate_model()

                for ode_method in self.configuration.ode_methods:
                    full_res = self.run_full(model, ode_method)

                    for stations_per_cell in self.configuration.stations_per_cell:
                        model.generate_cells(stations_per_cell)

                        #iter_res = self.run_iteration(model, ode_method)

                        #for iter_no, res in enumerate(iter_res[1]):
                        #    print(f"Accuracy score (iteration {iter_no+1}): {get_accuracy_score(full_res[0], full_res[1], iter_res[0], res)}")

                        for delta_t_ratio in self.configuration.delta_t_ratio:
                            delta_t = model.get_dt_from_ratio(delta_t_ratio)
                            disc_res = self.run_discrete(model, ode_method, delta_t)

                            print(f"Accuracy score: {get_accuracy_score(full_res[0], full_res[1], disc_res[0], disc_res[1])}")
        except:
            os.remove(self.output_folder + "output.csv")
            raise