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

import ctypes

import sklearn.cluster

import spatial_decomp_station
import spatial_decomp_strict

import comp
from oslo_data import *
import large_fluid

import pickle

from multiprocessing import Process, Lock, Manager
import multiprocessing.shared_memory

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
        self.strict_mu = None
        self.strict_phi = None


    def ode_export(self, ode_id):
        starting_values = [f"begin model ode{ode_id}", "begin init"]

        for station_id in range(self.n_stations):
            starting_values.append(f"x_{station_id} = {self.starting_bps[station_id]}")
        
        for start_stn in range(self.n_stations):
            for end_stn in range(self.n_stations):
                starting_values.append(f"y_{start_stn}_{end_stn} = 0")
        
        starting_values.append("end init")
        
        ode_equations = ["begin ODE"]

        for end_stn in range(self.n_stations):
            eq = [f"d(x_{end_stn}) = "]
            for start_stn in range(self.n_stations):
                if self.durations[start_stn][end_stn] == 0:
                    rate = 10000
                else:
                    rate = 1/self.durations[start_stn][end_stn]
                eq.append(f"{rate} * y_{start_stn}_{end_stn} + ")
            total_rate = sum(self.demands[end_stn])
            eq.append(f" -1 * {total_rate} * min(x_{end_stn}, 1)")

            ode_equations.append("".join(eq))


        for start_stn in range(self.n_stations):
            for end_stn in range(self.n_stations):
                eq = [f"d(y_{start_stn}_{end_stn}) = "]

                eq.append(f"{self.demands[start_stn][end_stn]} * min(x_{start_stn}, 1)")

                if self.durations[start_stn][end_stn] == 0:
                    rate = 10000
                else:
                    rate = 1/self.durations[start_stn][end_stn]
                eq.append(f" -1 * {rate} * y_{start_stn}_{end_stn}")

                ode_equations.append("".join(eq))

        ode_equations.append("end ODE")

        divider = ["////////////////////////////////////////////////////////////////////////", "// ODEs"]

        trailer = [f"reduceBDE(reducedFile=\"ode_{ode_id}_bde.ode\")",
                   f"reduceFDE(reducedFile=\"ode_{ode_id}_fde.ode\")",
                    "end model"]

        with open(f"ode_export/ode_{ode_id}.ode", "w+") as f:
            for line in starting_values + divider + ode_equations + trailer:
                f.write(line + "\n")
    
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

    def get_mu_strict(self):
        if not self.strict_mu:
            self.strict_mu  = [[[] for j in range(self.n_cells)] for i in range(self.n_cells)]
            self.strict_phi = [[[] for j in range(self.n_cells)] for i in range(self.n_cells)]

        return self.strict_mu

    def get_mu_strict(self):
        if not self.strict_phi:
            self.strict_mu  = [[[] for j in range(self.n_cells)] for i in range(self.n_cells)]
            self.strict_phi = [[[] for j in range(self.n_cells)] for i in range(self.n_cells)]

        return self.strict_phi
    
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
    
    def replicate(self, configuration):
        # create an identical model with different durations/demands
        out_model = ExperimentModel(
            self.stations,
            configuration.generate_durations(self.stations),
            configuration.generate_demands(self.stations),
            configuration.generate_bps(self.stations),
            self.seed
        )
        out_model.cell_to_station = self.cell_to_station
        out_model.station_to_cell = self.station_to_cell
        out_model.n_cells = self.n_cells

        return out_model


class ExperimentConfig:
    def __init__(self, seed, n_station_range, time_end):
        # meta
        self.repetitions_per_point = 50

        # parameters
        self.ode_methods             = ["RK45", "BDF"]
        self.stations_per_cell       = [5, 10, 15]
        self.delta_t_ratio           = [0.1, 0.05, 0.025, 0.0125, 0.00625] # setting delta T based on (x/[average rate])
        self.epsilon                 = [2, 1, 0.5, 0.25, 0.125]

        # random configuration
        self.n_station_range         = n_station_range
        self.x_location_range        = [0, 1]
        self.y_location_range        = [0, 1]
        self.station_demand_range    = [0, 0.05]
        #self.station_demand_range    = [0, 0.05]
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

def get_profit(t, y, n_stations, profit, subsidy_profit, subsidized_stations):
    y_station = y[:, -n_stations:]
    pass

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
                                #vectorized=True,
                                atol=ATOL,
                                method=ode_method)
        toc = time.perf_counter()
        print(f"Full Model finished, time: {toc-tic}.")
        return [[x for x in x_t.t], x_t.y, toc-tic]

    def run_pod(self, model, ode_method, full_solutions, n_modes):
        print("Running Proper Orthogonal Decomposition")
        tic = time.perf_counter()

        comp_model = comp.CompModel(model.durations, model.demands)

        U, Sigma, Vt = np.linalg.svd(full_solutions.T, full_matrices=False)

        U = U[:, :n_modes]
        Sigma = Sigma[:n_modes]
        Vt = Vt[:n_modes, :]

        A = np.diag(Sigma) @ Vt

        n_time_points = self.configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*self.configuration.time_end)/n_time_points for i in range(n_time_points+1)]

        starting_vector = [0 for i in range(comp_model.n_stations**2)] + model.starting_bps



        print(f"snapshot_svd: {snapshot_svd}")

        starting_vector = [0 for i in range(comp_model.n_stations**2)] + model.starting_bps

        toc = time.perf_counter()
        print(f"Proper Orthogonal Decomposition Finishd, time: {toc-tic}")
        return [None, x_arr, toc-tic, current_vector, traj_cells]

    def create_file(self, ext=""):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(f"{self.output_folder}output{ext}.csv", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["repetition", "n_stations", "method", "ode_method", "n_modes", "pod_time", "full_time", "sum_error", "max_error"])
    
    def write_row(self, row):
        with open(f"{self.output_folder}output.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run_validation(self):
        self.create_file()
        try:
            for repetition in range(self.configuration.repetitions_per_point):
                model = self.configuration.generate_model()

                #model.ode_export(repetition)
                
                n_stations = model.n_stations
                print(f"Repetition: {repetition}, n stations: {model.n_stations}")
                
                for ode_method in ["BDF", "RK45"]:
                    full_res = self.run_full(model, ode_method)
                    for n_modes in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                        pod_res = self.run_pod(model, ode_method, np.array(full_res[1]), n_modes)
                        sum_error = sum_accuracy(full_res[0], full_res[1], disc_res[0], disc_res[1])
                        max_error = max_accuracy(full_res[0], full_res[1], disc_res[0], disc_res[1])
                        self.write_row([repetition, n_stations, "pod", ode_method, n_modes, pod_res[2], full_res[2], sum_error, max_error])
                        
        except:
            shutil.rmtree(self.output_folder)
            raise
