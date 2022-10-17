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
RATE_MULTIPLIER = 1
HOURS = [11,12,13]

data_folder = "oslo_data_3"
out_folder = "oslo_out"

def get_cox_data(hour):
    durations = pd.read_csv(f"{data_folder}/cell_distances.csv")
    durations = durations[durations["hour"] == hour]

    n_cells = durations["start_cell"].max() + 1

    mu  = [[[] for end_cell in range(n_cells)] for start_cell in range(n_cells)]
    phi = [[[] for end_cell in range(n_cells)] for start_cell in range(n_cells)]

    # mu/phi format: [start_cell][end_cell][phase] => mu/phi
    for i, row in durations.iterrows():
        start_cell = int(row["start_cell"])
        end_cell   = int(row["end_cell"])

        if row["n"] == 1:
            mu[start_cell][end_cell]  = [float(row["lambda"])*RATE_MULTIPLIER]
            phi[start_cell][end_cell] = [1.0]
        elif row["n"] == 2:
            mu[start_cell][end_cell]  = [float(row["mu1"]) * RATE_MULTIPLIER, float(row["mu2"]) * RATE_MULTIPLIER]
            phi[start_cell][end_cell] = [float(row["phi1"]), 1.0]
        else:
            erlang_phases = int(row["n"])
            lam = float(row["lambda"]) * RATE_MULTIPLIER
            mu[start_cell][end_cell]  = [lam for i in range(erlang_phases)]
            phi[start_cell][end_cell] = [0.0 for i in range(erlang_phases-1)] + [1.0]

    return [mu, phi]

def get_demands(hour, cell_to_station, station_to_cell):
    in_demands_frame = pd.read_csv(f"{data_folder}/in_demands.csv")
    out_demands_frame = pd.read_csv(f"{data_folder}/out_demands.csv")
    in_probabilities_frame = pd.read_csv(f"{data_folder}/in_probs.csv")

    in_demands_frame = in_demands_frame[in_demands_frame["hour"] == hour]
    out_demands_frame = out_demands_frame[out_demands_frame["hour"] == hour]
    in_probabilities_frame = in_probabilities_frame[in_probabilities_frame["hour"] == hour]

    n_cells = in_demands_frame["start_cell"].max() + 1

    # end_cell => start_cell => station => demand
    in_demands = [[[0 for station in cell_to_station[end_cell]] for start_cell in range(n_cells)] for end_cell in range(n_cells)]

    for i, row in in_demands_frame.iterrows():
        start_cell = int(row["start_cell"])
        end_station = int(row["end"])
        end_cell = station_to_cell[end_station]
        end_station = cell_to_station[end_cell].index(end_station)
        
        in_demands[end_cell][start_cell][end_station] = float(row["hourly_demand"])


    # end_cell => start_cell => station => probability
    in_probabilities = [[[0 for station in cell_to_station[end_cell]] for start_cell in range(n_cells)] 
                    for end_cell in range(n_cells)]

    for i, row in in_probabilities_frame.iterrows():
        start_cell = int(row["start_cell"])
        end_station = int(row["end"])
        end_cell = station_to_cell[end_station]
        end_station = cell_to_station[end_cell].index(end_station)

        in_probabilities[end_cell][start_cell][end_station] = float(row["prob"])
    


    # start_cell => station => end_cell => demand
    out_demands = [[[0 for end_cell in range(n_cells)] for station in cell_to_station[start_cell]] for start_cell in range(n_cells)]

    for i, row in out_demands_frame.iterrows():
        start_station = int(row["start"])
        start_cell = station_to_cell[start_station]
        end_cell = int(row["end_cell"])
        start_station = cell_to_station[start_cell].index(start_station)
        

        out_demands[start_cell][start_station][end_cell] = float(row["hourly_demand"])



    return [in_demands, in_probabilities, out_demands]

def run_simulation(model_data, n_runs, current_vector="none"):
    print("Running simulation..")

    tic = time.perf_counter()

    current_vector_list = []
    hr_vector_list = []

    for run in range(n_runs):
        n_entries = model_data.n_cells**2 + model_data.n_stations

        x_arr = np.array([])

        delay_phase_ct = [sum([len(x) for x in model_data.mu[i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
        
        delay_offset = [] # start_cell -> end_cell->idx
        station_offset = [] # start_cell -> idx of station 0

        for start_cell in range(model_data.n_cells):
            sc_delay_offset = []
            cur_offset = 0

            for end_cell in range(model_data.n_cells):
                sc_delay_offset.append(cur_offset)
                cur_offset += len(model_data.mu[start_cell][end_cell])

            delay_offset.append(sc_delay_offset)
            station_offset.append(cur_offset)
            

        if current_vector == "none":
            current_vector = [
                [0.0 for i in range(delay_phase_ct[cell_idx])] +
                [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
                    for cell_idx in range(model_data.n_cells)
            ]
        
        time_points = []

        t = 0

        
        while t < model_data.time_end:
            total_rate = 0
            for start_cell, sc_data in enumerate(model_data.mu):
                for end_cell, ec_data in enumerate(sc_data):
                    for phase_no, mu_val in enumerate(ec_data):
                        phase_rate = mu_val * current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no]
                        total_rate += phase_rate

            for start_cell, sc_data in enumerate(model_data.out_demands):
                for station_idx, st_data in enumerate(sc_data):
                    for end_cell, lam_val in enumerate(st_data):
                        dep_rate = lam_val if current_vector[start_cell][station_offset[start_cell] + station_idx] >= 1 else 0
                        total_rate += dep_rate
            
            cdf_val = random.random()
            c_prob = 0
            found_event = False

            for start_cell, sc_data in enumerate(model_data.mu):
                for end_cell, ec_data in enumerate(sc_data):
                    for phase_no, mu_val in enumerate(ec_data):
                        phase_rate = mu_val * current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no]
                        c_prob += (phase_rate/total_rate)

                        if c_prob >= cdf_val:
                            found_event = True

                            if random.random() <= model_data.phi[start_cell][end_cell][phase_no]:
                                # departure
                                current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no] -= 1

                                station_cdf_val = random.random()
                                station_c_prob = 0

                                for station_idx, station_prob in enumerate(model_data.in_probabilities[end_cell][start_cell]):
                                    station_c_prob += station_prob
                                    if station_c_prob >= station_cdf_val:
                                        current_vector[end_cell][station_offset[end_cell] + station_idx] += 1
                                        break

                            else:
                                # update next phase
                                current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no] -= 1
                                current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no + 1] += 1

                            break
                    if found_event:
                        break
                if found_event:
                    break

            if not found_event:
                for start_cell, sc_data in enumerate(model_data.out_demands):
                    for station_idx, st_data in enumerate(sc_data):
                        for end_cell, lam_val in enumerate(st_data):
                            dep_rate = lam_val if current_vector[start_cell][station_offset[start_cell] + station_idx] >= 1 else 0

                            c_prob += (dep_rate/total_rate)

                            if c_prob >= cdf_val:
                                # depart and add to delay at first phase 
                                found_event = True
                                current_vector[start_cell][station_offset[start_cell] + station_idx] -= 1
                                current_vector[start_cell][delay_offset[start_cell][end_cell] + phase_no] += 1
                                break

                        if found_event:
                            break
                    if found_event:
                        break

            t += -math.log(random.random())/total_rate
        current_vector_list.append(current_vector)

    current_vector_mean = copy.deepcopy(current_vector)

    for cell_idx, cell_vec in enumerate(current_vector):
        for stn_id, stn_val in enumerate(cell_vec):
            current_vector_mean[cell_idx][stn_id] = sum([run_vec[cell_idx][stn_id] for run_vec in current_vector_list])/n_runs

    toc = time.perf_counter()
    print(f"Simulation finished, time: {toc-tic}")
    return [time_points, x_arr, toc-tic, current_vector_mean]


def get_starting_bps():
    initial_loads = pd.read_csv(f"{data_folder}/initial_loads.csv")
    return list(initial_loads["start_level"])

def run_discrete(model_data, traj_cells, ode_method, step_size, inflate=True, current_vector="none"):
    print("Running Discrete-Step Submodeling")

    tic = time.perf_counter()

    n_entries = model_data.n_cells**2 + model_data.n_stations

    x_arr = np.array([])

    delay_phase_ct = [sum([len(x) for x in mu[i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    
    
    trajectories = [[0 for j in range(traj_cells[end_cell].in_offset)] for end_cell in range(model_data.n_cells)]

    if current_vector == "none":
        current_vector = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [float(x) for i, x in enumerate(starting_bps) if i in list(cell_to_station[cell_idx])]
                for cell_idx in range(model_data.n_cells)
        ]
        """current_vector = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [model_data.const_bike_load for i, x in enumerate(starting_bps) if i in list(cell_to_station[cell_idx])]
                for cell_idx in range(model_data.n_cells)
        ]"""
    station_vals = [[] for i in range(model_data.n_stations)]

    time_points = []

    t = 0

    n_bikes = sum([sum(x) for x in current_vector])

    
    while t < model_data.time_end:
        print(f"t: {t}")
        sub_time_points = [t+(i*(step_size/model_data.steps_per_dt)) for i in range(model_data.steps_per_dt)]

        if len(sub_time_points) == 0:
            sub_time_points.append(t)

        time_points += sub_time_points
        
        new_trajectories = copy.deepcopy(trajectories)
        new_vector = copy.deepcopy(current_vector)


        x_iter = np.array([[0.0 for x in range(len(sub_time_points))] for i in range(n_entries)])

        for cell_idx in range(model_data.n_cells):
            if cell_idx % 20 == 0:
                print(f"Cell: {cell_idx}")
            traj_cells[cell_idx].set_trajectories(trajectories[cell_idx])
            #print(trajectories[cell_idx])

            x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_const, [t, t+step_size], current_vector[cell_idx], 
                                    t_eval = sub_time_points,
                                    method=ode_method, atol=ATOL)


            x_iter = np.concatenate([x_iter, x_t.y])
            
            for i, station_id in enumerate(traj_cells[cell_idx].stations):
                sy_idx = delay_phase_ct[cell_idx]+i
                current_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])
                
            for next_cell in range(model_data.n_cells):
                for phase, phase_rate in enumerate(mu[cell_idx][next_cell]):
                    phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                    
                    last_val = float(x_t.y[phase_qty_idx, -1])
                    current_vector[cell_idx][phase_qty_idx] = last_val
                    new_trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase] = last_val

        if inflate:
            end_bikes = sum([sum(x) for x in new_vector])
            # do loss by number of bikes in delays
            loss_ptg = float(n_bikes-end_bikes)/n_bikes
            inflation_factor = 1/(1-loss_ptg)
            for start_stn in range(model.n_stations):
                for end_stn in range(model.n_stations):
                    new_lstates[start_stn][end_stn] *= inflation_factor
            for cell_idx in range(model.n_cells):
                new_vector[cell_idx] = [x * inflation_factor for x in new_vector[cell_idx]]


        if x_arr.size == 0:
            x_arr = x_iter
        else:
            x_arr = np.concatenate([x_arr, x_iter], axis=1)

        trajectories = new_trajectories
        t += step_size


    toc = time.perf_counter()
    print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
    return [time_points, x_arr, toc-tic, current_vector]

class ModelData:
    def __init__(self, n_stations, n_cells, starting_bps, mu, phi, in_demands, in_probabilities, out_demands):
        self.n_stations = n_stations
        self.n_cells = n_cells
        self.starting_bps = starting_bps
        self.mu = mu
        self.phi = phi
        self.in_demands = in_demands
        self.in_probabilities = in_probabilities
        self.out_demands = out_demands
        self.time_end = 4
        self.steps_per_dt = 100
        self.const_bike_load = 6

def run_hour(hour):
    # demands
    in_demands, in_probabilities, out_demands = get_demands(hour,cell_to_station, station_to_cell)

    # phase processes
    mu, phi = get_cox_data(hour)

    # build StrictTrajCellCox for each station cell
    traj_cells = [
        spatial_decomp_strict.StrictTrajCellCox(cell_idx, cell_to_station[cell_idx], 
            mu, phi, in_demands[cell_idx], in_probabilities[cell_idx], 
            out_demands[cell_idx])

            for cell_idx in range(n_cells)
    ]

    # run model
    starting_bps = get_starting_bps()
    model_data = ModelData(n_stations, n_cells, starting_bps, mu, phi, in_demands, in_probabilities, out_demands)
    time_points, x_arr, time_val, last_vector_sim = run_simulation(model_data,6)
    time_points, x_arr, time_val, last_vector = run_discrete(model_data, traj_cells, "RK45", 0.5) # HOOK
 
    station_res = [0 for i in range(n_stations)]
    hourly_res = [[0 for i in range(n_stations)] for hr in HOURS]
    station_simres = [0 for i in range(n_stations)]
    hourly_simres = [[0 for i in range(n_stations)] for hr in HOURS]

    cell_start = 0

    total_bikes = 0

    for cell_idx, traj_cell in enumerate(traj_cells):
        total_bikes += sum(last_vector[cell_idx])
        for station_cell_idx, station_idx in enumerate(cell_to_station[cell_idx]):
            station_res[station_idx] = last_vector[cell_idx][traj_cell.station_offset + station_cell_idx]
            station_simres[station_idx] = last_vector_sim[cell_idx][traj_cell.station_offset + station_cell_idx]

    with open(f"{out_folder}/res_{hour}", "w") as f:
        json.dump(station_res, f)
            
    with open(f"{out_folder}/res_sim_{hour}", "w") as f:
        json.dump(station_simres, f)
    



if __name__ == "__main__":
    # stations
    stations = pd.read_csv(f"{data_folder}/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = stations["cell"].max() + 1
    n_stations = stations["index"].max() + 1
    cell_to_station = [[] for i in range(n_cells)]
    station_to_cell = [0 for i in range(n_stations)]
    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])
    
    run_hour(10)

    """with open ("oslo_hr", "w") as f:
        json.dump(hourly_res, f)
    with open ("oslo_final", "w") as f:
        json.dump(station_res, f)

    with open ("oslo_hr_sim", "w") as f:
        json.dump(hourly_simres, f)
    with open ("oslo_final_sim", "w") as f:
        json.dump(station_simres, f)"""
