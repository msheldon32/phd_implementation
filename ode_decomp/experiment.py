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
import shutil

import sklearn.cluster

import spatial_decomp_station
import spatial_decomp_strict

import comp
from oslo_data import *
import large_fluid

import pickle

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

        self.strict_durations = None
        self.strict_in_demands = None
        self.strict_out_demands = None
        self.strict_in_probs = None
    
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
    
    def get_durations_strict(self):
        if not self.strict_durations:
            self.strict_durations = [[-1 for j in range(self.n_cells)] for i in range(self.n_cells)]

            total_acc = 0
            total_N = 0

            for start_cell in range(self.n_cells):
                for end_cell in range(self.n_cells):
                    acc = 0
                    N = 0
                    for srt_stn in self.cell_to_station[start_cell]:
                        for end_stn in self.cell_to_station[end_cell]:
                            if np.isnan(self.durations[srt_stn][end_stn]) or np.isnan(self.demands[srt_stn][end_stn]):
                                continue
                            acc += self.durations[srt_stn][end_stn]*self.demands[srt_stn][end_stn]
                            total_acc += self.durations[srt_stn][end_stn]*self.demands[srt_stn][end_stn]
                            N += self.demands[srt_stn][end_stn]
                            total_N += self.demands[srt_stn][end_stn]
                    if N > 0:
                        self.strict_durations[start_cell][end_cell] = acc/N

            for start_cell in range(self.n_cells):
                for end_cell in range(self.n_cells):
                    if self.strict_durations[start_cell][end_cell] == -1:
                        # use the overall average if there's no information available
                        self.strict_durations[start_cell][end_cell] = total_acc/total_N

        return self.strict_durations
    
    def get_in_demands_strict(self):
        if not self.strict_in_demands:
            self.strict_in_demands = []
            # for each cell,
            # find the total_demand from each cell to each station
            for cell_idx in range(self.n_cells):
                in_demands_cell = [[0 for i in range(len(self.cell_to_station[cell_idx]))] for j in range(self.n_cells)]
                for start_cell in range(self.n_cells):
                    for end_idx, end_stn in enumerate(self.cell_to_station[cell_idx]):
                        total_demand = 0
                        for srt_stn in self.cell_to_station[start_cell]:
                            total_demand += self.demands[srt_stn][end_stn]
                        in_demands_cell[start_cell][end_idx] = total_demand

                        assert not np.isnan(total_demand)

                self.strict_in_demands.append(in_demands_cell)
        return self.strict_in_demands
    
    def get_out_demands_strict(self):
        if not self.strict_out_demands:
            self.strict_out_demands = []
            # for each cell,
            # find the total_demand from each station to each cell
            for cell_idx in range(self.n_cells):
                out_demands_cell = [[0 for i in range(self.n_cells)] for j in range(len(self.cell_to_station[cell_idx]))]
                for end_cell in range(self.n_cells):
                    for srt_idx, srt_stn in enumerate(self.cell_to_station[cell_idx]):
                        total_demand = 0
                        for end_stn in self.cell_to_station[end_cell]:
                            total_demand += self.demands[srt_stn][end_stn]
                        out_demands_cell[srt_idx][end_cell] = total_demand

                        assert not np.isnan(total_demand)
                self.strict_out_demands.append(out_demands_cell)
        return self.strict_out_demands
    
    def get_in_probs_strict(self):
        if not self.strict_in_probs:
            self.strict_in_probs = []

            in_demands = self.get_in_demands_strict()

            for cell_idx in range(self.n_cells):
                in_probs_cell = [[0 for i in range(len(self.cell_to_station[cell_idx]))] for j in range(self.n_cells)]

                for start_cell in range(self.n_cells):
                    total_demand = sum(in_demands[cell_idx][start_cell])

                    if total_demand == 0:
                        in_probs_cell[start_cell][0] = 1
                        continue

                    for end_idx, end_stn in enumerate(self.cell_to_station[cell_idx]):
                        in_probs_cell[start_cell][end_idx] = in_demands[cell_idx][start_cell][end_idx] / total_demand
                    assert not np.isnan(in_probs_cell[start_cell][end_idx])

                self.strict_in_probs.append(in_probs_cell)
        return self.strict_in_probs
    
    def reset_strict(self):
        self.strict_durations = None
        self.strict_in_demands = None
        self.strict_out_demands = None
        self.strict_in_probs = None


class ExperimentConfig:
    def __init__(self, seed, n_station_range, time_end):
        # meta
        self.repetitions_per_point = 50

        # parameters
        self.ode_methods             = ["RK45", "BDF"]
        self.stations_per_cell       = [5, 10, 15]
        self.delta_t_ratio           = [0.1, 0.05, 0.025] # setting delta T based on (x/[average rate])
        self.epsilon                 = [2, 1, 0.5]

        # random configuration
        self.n_station_range         = n_station_range
        self.x_location_range        = [0, 1]
        self.y_location_range        = [0, 1]
        self.station_demand_range    = [0, 0.5]
        self.noise_moments_distance  = [0, 0.2]
        self.bps_range               = [0, 15]

        # constants
        self.max_iterations = 100
        self.iter_tolerance = 1
        self.time_end       = time_end
        self.min_duration   = 0.01
        self.steps_per_dt   = 100

        self.seed = seed

    
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
                distance = euclidian_distance(src_stn, dst_stn)
                eps = generate_gaussian(self.noise_moments_distance)
                demands[i][j] = generate_uniform(self.station_demand_range)
        
        return demands

    def generate_bps(self, stations):
        n_stations = len(stations)

        bps = [float(generate_dis_uniform(self.bps_range)) for i in range(n_stations)]
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

def sum_accuracy(base_t, base_y, alt_t, alt_y):
    diff = abs(refit_times(base_t, base_y, alt_t) - alt_y)
    denom = base_y[:,0].sum()*len(alt_t)*2
    #return (diff/total_val).max()
    return (diff/denom).sum()

def max_accuracy(base_t, base_y, alt_t, alt_y):
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


    
    def run_iteration(self, model, ode_method, epsilons, h_analysis=False):
        print("Running Trajectory-Based Iteration")

        all_res = []

        tic = time.perf_counter()

        n_entries = model.n_stations**2 + model.n_stations

        cur_epsilon = 0

        x_res = []

        h_cells = {}
        h_time  = 0

        starting_vector = [
            [0 for i in range(len(model.cell_to_station[cell_idx]) * model.n_stations)] +
            [x for i, x in enumerate(model.starting_bps) if i in list(model.cell_to_station[cell_idx])]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_station.TrajCell(cell_idx, model.station_to_cell, model.cell_to_station, 
                                    sorted(list(model.cell_to_station[cell_idx])), model.durations, model.demands)
                            for cell_idx in range(model.n_cells)]

        n_time_points = self.configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*self.configuration.time_end)/n_time_points for i in range(n_time_points+1)]
        trajectories = [[np.zeros(n_time_points+1) for j in range(model.n_stations)] for i in range(model.n_stations)]
        
        for traj_cell in traj_cells:
            traj_cell.set_timestep(self.configuration.time_end/n_time_points)

        station_vals = []

        n_iterations = 0

        non_h_idx = []

        if h_analysis:
            non_h_idx = np.array([True for i in range(n_entries)])

        for iter_no in range(self.configuration.max_iterations):
            n_iterations = iter_no + 1

            for tc in traj_cells:
                tc.set_trajectories(trajectories)

            new_trajectories = copy.deepcopy(trajectories)

            x_iter = np.array([[0.0 for i in range(n_time_points+1)] for i in range(n_entries)])

            for cell_idx in range(model.n_cells):
                if h_analysis and cell_idx in h_cells:
                    x_iter[traj_cells[cell_idx].get_idx(), :] = 0.0
                    continue

                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array, [0, self.configuration.time_end], starting_vector[cell_idx], 
                                        t_eval=time_points, 
                                        method=ode_method, atol=ATOL)
                

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    sy_idx = traj_cells[cell_idx].get_station_idx(i)
                    
                    for dst_stn in range(model.n_stations):
                        y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        new_trajectories[src_stn][dst_stn] = x_t.y[y_idx, :]

                if h_analysis:
                    s_ct = traj_cells[cell_idx].s_in_cell
                    
                    if not (x_t.y[-s_ct:, :] < 1).any():
                        print(f"added h cell {cell_idx}")

                        h_cells[cell_idx] = iter_no
                        x_res[-1][traj_cells[cell_idx].get_idx(), :] = 0.0
                        x_iter[traj_cells[cell_idx].get_idx(), :] = 0.0

                        non_h_idx[traj_cells[cell_idx].get_idx()] = False

                        continue
                    if not (x_t.y[-traj_cells[cell_idx].s_in_cell:, :] > 1).any():
                        print(f"Found l cell {cell_idx}")
                
                x_iter[traj_cells[cell_idx].get_idx(), :] = x_t.y
            
            x_res.append(x_iter)

            trajectories = new_trajectories

            if iter_no > 0:
                if h_analysis:
                    error_score = (abs(x_res[-1][non_h_idx,:] - x_res[-2][non_h_idx,:])).max()
                else:
                    error_score = (abs(x_res[-1] - x_res[-2])).max()

                h_rec = 0

                if h_analysis and cur_epsilon < len(epsilons):
                    h_tic  = time.perf_counter()

                    # reconcile all h cells
                    for cell_idx in h_cells.keys():
                        x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array, [0, self.configuration.time_end], starting_vector[cell_idx], 
                                                t_eval=time_points, 
                                                method=ode_method, atol=ATOL)

                        x_iter[traj_cells[cell_idx].get_idx(), :] = x_t.y
                        

                        for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                            sy_idx = traj_cells[cell_idx].get_station_idx(i)
                            
                            for dst_stn in range(model.n_stations):
                                y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                                new_trajectories[src_stn][dst_stn] = x_t.y[y_idx, :]

                    h_toc  = time.perf_counter()
                    h_rec  = h_toc - h_tic
                    h_time += h_rec

                while cur_epsilon < len(epsilons):
                    if error_score < epsilons[cur_epsilon]:
                        toc = time.perf_counter()
                        all_res.append([time_points, x_res[-1], toc-tic-h_time + h_rec, n_iterations])
                        cur_epsilon += 1
                    else:
                        break
                
                print(f"Iteration complete, time: {time.perf_counter()-tic}, error: {error_score}")
                if cur_epsilon >= len(epsilons):
                    break

        print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

        return all_res
    
    def run_discrete(self, model, ode_method, step_size):
        print("Running Discrete-Step Submodeling")

        tic = time.perf_counter()

        n_entries = model.n_stations**2 + model.n_stations

        x_arr = np.array([])

        starting_vector = [
            [0 for i in range(len(model.cell_to_station[cell_idx]) * model.n_stations)] +
            [x for i, x in enumerate(model.starting_bps) if i in list(model.cell_to_station[cell_idx])]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_station.TrajCell(cell_idx, model.station_to_cell, model.cell_to_station, 
                                    sorted(list(model.cell_to_station[cell_idx])), model.durations, model.demands)
                            for cell_idx in range(model.n_cells)]

        last_states = [[0 for j in range(model.n_stations)] for i in range(model.n_stations)]

        station_vals = [[] for i in range(model.n_stations)]

        time_points = []
        
        current_vector = copy.deepcopy(starting_vector)

        t = 0

        
        while t < self.configuration.time_end:
            print(f"t: {t}") # HOOK
            sub_time_points = [t+(i*(step_size/self.configuration.steps_per_dt)) for i in range(self.configuration.steps_per_dt)]
            if len(sub_time_points) == 0:
                sub_time_points.append(t)
            time_points += sub_time_points
            
            new_lstates = copy.deepcopy(last_states)
            new_vector = copy.deepcopy(current_vector)


            x_iter = np.array([[0.0 for x in range(len(sub_time_points))] for i in range(n_entries)])
            

            for cell_idx in range(model.n_cells):
                traj_cells[cell_idx].set_last_states(last_states)

                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_const, [t, t+step_size], current_vector[cell_idx], 
                                        t_eval = sub_time_points,
                                        method=ode_method, atol=ATOL)

                x_iter[traj_cells[cell_idx].get_idx(), :] = x_t.y

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    sy_idx = traj_cells[cell_idx].get_station_idx(i)
                    
                    station_vals[src_stn] += [comp.get_traj_fn(x_t.t, x_t.y[sy_idx,:])(t) for t in sub_time_points]
                    
                    new_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

                    for dst_stn in range(model.n_stations):
                        y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        last_val = float(x_t.y[y_idx, -1])
                        new_vector[cell_idx][y_idx] = last_val
                        delay_rate = 1/model.durations[src_stn][dst_stn]
                        new_lstates[src_stn][dst_stn] = last_val

            if x_arr.size == 0:
                x_arr = x_iter
            else:
                x_arr = np.concatenate([x_arr, x_iter], axis=1)

            last_states = new_lstates
            current_vector = new_vector

            t += step_size
        toc = time.perf_counter()
        print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
        return [time_points, x_arr, toc-tic]
    
    def run_iteration_strict(self, model, ode_method, epsilons):
        print("Running Trajectory-Based Iteration")

        all_res = []

        tic = time.perf_counter()

        n_entries = model.n_cells**2 + model.n_stations

        cur_epsilon = 0

        x_res = []

        starting_vector = [
            [0.0 for i in range(model.n_cells)] +
            [float(x) for i, x in enumerate(model.starting_bps) if i in list(model.cell_to_station[cell_idx])]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_strict.StrictTrajCell(cell_idx, sorted(list(model.cell_to_station[cell_idx])), 
                                    model.get_durations_strict(),
                                    model.get_in_demands_strict()[cell_idx], 
                                    model.get_in_probs_strict()[cell_idx], model.get_out_demands_strict()[cell_idx])
                            for cell_idx in range(model.n_cells)]

        n_time_points = self.configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*self.configuration.time_end)/n_time_points for i in range(n_time_points+1)]
        trajectories = [[np.zeros(n_time_points+1) for j in range(model.n_cells)] for i in range(model.n_cells)]
        
        for traj_cell in traj_cells:
            traj_cell.set_timestep(self.configuration.time_end/n_time_points)

        station_vals = []

        n_iterations = 0

        for iter_no in range(self.configuration.max_iterations):
            n_iterations = iter_no + 1

            for i,tc in enumerate(traj_cells):
                tc.set_trajectories(trajectories[i])

            new_trajectories = copy.deepcopy(trajectories)

            total_bikes = 0

            x_iter = np.array([[0.0 for i in range(n_time_points+1)] for i in range(n_entries)])
            
            for cell_idx in range(model.n_cells):
                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array, [0, self.configuration.time_end], starting_vector[cell_idx], 
                                        t_eval=time_points, 
                                        method=ode_method, atol=ATOL)

                x_iter[traj_cells[cell_idx].get_idx(), :] = x_t.y
                

                for dst_cell in range(model.n_cells):
                    new_trajectories[dst_cell][cell_idx] = x_t.y[dst_cell, :]
                total_bikes += sum(x_t.y[:,-1])
            
            x_res.append(x_iter)

            trajectories = new_trajectories

            if iter_no > 0:
                error_score = (abs(x_res[-1] - x_res[-2])).max()
                while cur_epsilon < len(epsilons):
                    if error_score < epsilons[cur_epsilon]:
                        toc = time.perf_counter()
                        all_res.append([time_points, x_res[-1], toc-tic, n_iterations])
                        cur_epsilon += 1
                    else:
                        break
                
                if cur_epsilon >= len(epsilons):
                    break
                print(f"Iteration complete, time: {time.perf_counter()-tic}, error: {error_score}")

        print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

        return all_res
    
    def run_discrete_strict(self, model, ode_method, step_size):
        print("Running Discrete-Step Submodeling")

        tic = time.perf_counter()

        n_entries = model.n_cells**2 + model.n_stations

        x_arr = np.array([])

        current_vector = [
            [0.0 for i in range(model.n_cells)] +
            [float(x) for i, x in enumerate(model.starting_bps) if i in list(model.cell_to_station[cell_idx])]
                for cell_idx in range(model.n_cells)
        ]
                
        traj_cells = [spatial_decomp_strict.StrictTrajCell(cell_idx, sorted(list(model.cell_to_station[cell_idx])), 
                                    model.get_durations_strict(),
                                    model.get_in_demands_strict()[cell_idx], 
                                    model.get_in_probs_strict()[cell_idx], model.get_out_demands_strict()[cell_idx])
                            for cell_idx in range(model.n_cells)]

        trajectories = [[0 for j in range(model.n_cells)] for i in range(model.n_cells)]

        station_vals = [[] for i in range(model.n_stations)]

        time_points = []

        t = 0

        
        while t < self.configuration.time_end:
            print(f"t: {t}") # HOOK
            sub_time_points = [t+(i*(step_size/self.configuration.steps_per_dt)) for i in range(self.configuration.steps_per_dt)]

            if len(sub_time_points) == 0:
                sub_time_points.append(t)

            time_points += sub_time_points
            
            new_trajectories = copy.deepcopy(trajectories)
            new_vector = copy.deepcopy(current_vector)


            x_iter = np.array([[0.0 for x in range(len(sub_time_points))] for i in range(n_entries)])

            for cell_idx in range(model.n_cells):
                traj_cells[cell_idx].set_trajectories(trajectories[cell_idx])

                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_const, [t, t+step_size], current_vector[cell_idx], 
                                        t_eval = sub_time_points,
                                        method=ode_method, atol=ATOL)

                x_iter[traj_cells[cell_idx].get_idx(), :] = x_t.y

                for i, station_id in enumerate(traj_cells[cell_idx].stations):
                    sy_idx = model.n_cells+i
                    current_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

                for dst_cell in range(model.n_cells):
                    last_val = float(x_t.y[dst_cell, -1])
                    new_trajectories[dst_cell][cell_idx] = last_val

            if x_arr.size == 0:
                x_arr = x_iter
            else:
                x_arr = np.concatenate([x_arr, x_iter], axis=1)

            trajectories = new_trajectories

            t += step_size
        toc = time.perf_counter()
        print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
        return [time_points, x_arr, toc-tic]
    
    def write_row(self, row):
        with open(self.output_folder + "output.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def create_file(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(self.output_folder + "output.csv", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "solution_method", "spatial_simplification", "ode_method", "stations_per_cell", 
                             "delta_t_ratio_or_epsilon", "delta_t_or_n_iterations", "time", "std_time", "error_sum_metric", "error_max_metric"])
    
    def save_model(self, repetition, model):
        with open(self.output_folder + "model_{}".format(repetition), "xb") as f:
            pickle.dump(model, f)


    def run(self):
        self.create_file()
        try:
            for repetition in range(self.configuration.repetitions_per_point):
                model = self.configuration.generate_model()
                n_stations = model.n_stations
                print(f"Repetition: {repetition}, n stations: {model.n_stations}")

                self.save_model(repetition, model)

                for ode_method in self.configuration.ode_methods:
                    full_res = self.run_full(model, ode_method)


                    for stations_per_cell in self.configuration.stations_per_cell:
                        model.reset_strict()
                        model.generate_cells(stations_per_cell)
                        
                        h_res = self.run_iteration(model, ode_method, self.configuration.epsilon, h_analysis=True)

                        for epsilon, iter_res in zip(self.configuration.epsilon, h_res):
                            sum_error = sum_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            max_error = max_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            self.write_row([repetition, n_stations, "traj_iteration", "h", ode_method, stations_per_cell, epsilon, iter_res[3], iter_res[2], full_res[2], sum_error, max_error])
                        
                        strict_res = self.run_iteration_strict(model, ode_method, self.configuration.epsilon)

                        for epsilon, iter_res in zip(self.configuration.epsilon, strict_res):
                            sum_error = sum_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            max_error = max_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            self.write_row([repetition, n_stations, "traj_iteration", "strict", ode_method, stations_per_cell, epsilon, iter_res[3], iter_res[2], full_res[2], sum_error, max_error])
                        #HOOK
                        all_res = self.run_iteration(model, ode_method, self.configuration.epsilon)

                        for epsilon, iter_res in zip(self.configuration.epsilon, all_res):
                            sum_error = sum_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            max_error = max_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], iter_res[0], iter_res[1][(-model.n_stations):,:])
                            self.write_row([repetition, n_stations, "traj_iteration", "none", ode_method, stations_per_cell, epsilon, iter_res[3], iter_res[2], full_res[2], sum_error, max_error])


                        for delta_t_ratio in self.configuration.delta_t_ratio:
                            delta_t = model.get_dt_from_ratio(delta_t_ratio)

                            disc_res_strict = self.run_discrete_strict(model, ode_method, delta_t)
                            sum_error = sum_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], disc_res_strict[0], disc_res_strict[1][(-model.n_stations):,:])
                            max_error = max_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], disc_res_strict[0], disc_res_strict[1][(-model.n_stations):,:])
                            self.write_row([repetition, n_stations, "discrete_step", "strict", ode_method, stations_per_cell, delta_t_ratio, delta_t, disc_res_strict[2], full_res[2], sum_error, max_error])

                            disc_res = self.run_discrete(model, ode_method, delta_t)
                            sum_error = sum_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], disc_res[0], disc_res[1][(-model.n_stations+1):,:])
                            max_error = max_accuracy(full_res[0], full_res[1][(-model.n_stations):,:], disc_res[0], disc_res[1][(-model.n_stations):,:])
                            self.write_row([repetition, n_stations, "discrete_step", "none", ode_method, stations_per_cell, delta_t_ratio, delta_t, disc_res[2], full_res[2], sum_error, max_error])
                        
        except:
            shutil.rmtree(self.output_folder)
            raise
