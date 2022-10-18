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

from multiprocessing import Process, Lock, Manager
import multiprocessing.shared_memory

import gc

import ctypes

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

def run_iteration(model_data, traj_cells, ode_method, step_size, cell_limit=False, current_vector="none", prior_iterations ="none"):
    print(f"Running Trajectory-Based Iteration (parallelized, limit: {cell_limit})")
    all_res = []

    tic = time.perf_counter()

    n_entries = model_data.n_stations**2 + model_data.n_stations
    x_res = []

    if prior_iterations != "none":
        x_res = prior_iterations

    h_cells = {}
    
    limited_cells = Manager().dict()
    h_time  = 0
    
    if current_vector == "default":
        current_vector = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
                for cell_idx in range(model_data.n_cells)
        ]
        
    n_time_points = int(model_data.time_end*TIME_POINTS_PER_HOUR)
    time_points = [(i*model_data.time_end)//n_time_points for i in range(n_time_points+1)]
    trajectories = [[np.zeros(n_time_points+1) for j in range(model_data.n_stations)] for i in range(model_data.n_stations)]
    
    for traj_cell in traj_cells:
        traj_cell.set_timestep(model_data.time_end/n_time_points)

    station_vals = []

    n_iterations = 0

    non_h_idx = []

    last_iter = False

    for iter_no in range(model_data.max_iterations):
        n_iterations = iter_no + 1

        new_trajectories = copy.deepcopy(trajectories)

        gc.collect()

        print((model_data.n_stations ** 2) * (n_time_points + 1))
        iwrap = multiprocessing.Array(ctypes.c_float, int(n_entries*(n_time_points+1)))
        twrap = [[multiprocessing.Array(ctypes.c_float, n_time_points+1) for j in range(model_data.n_stations)] for i in range(model_data.n_stations)]


        cell_threads = [None for i in range(model_data.n_cells)]
        x_t_cell = [None for i in range(model_data.n_cells)]

        x_iter_shape = (n_entries, n_time_points + 1)

        finished_cells = Manager.dict()

        for cell_idx in range(model_data.n_cells):
            lc_lock = Lock()
            xiterlock = Lock()

            def limited_cell_fn(cell_idx):#, trajwrap, iterwrap, lc_lock, xiterlock):
                #x_iter = np.frombuffer(iterwrap.get_obj()).reshape(x_iter_shape)
                cell_indices = traj_cells[cell_idx].get_idx()
                if len(x_res) != 0:
                    np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[cell_indices, :] = x_res[-1][cell_indices, :]
                else:
                    np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[cell_indices, :] = 0

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    for dst_stn in range(model_data.n_stations):
                        np.frombuffer(twrap[src_stn][dst_stn].get_obj())[:][:] = trajectories[src_stn][dst_stn]
                finished_cells[cell_idx] = True

            def cell_fn(cell_idx):#, trajwrap, iterwrap):#, lc_lock, xiterlock, limited_cells):
                traj_cells[cell_idx].set_trajectories(trajectories)

                #cell_indices = traj_cells[cell_idx].get_idx()

                x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array, [0, self.configuration.time_end], current_vector[cell_idx], 
                                        t_eval=time_points, 
                                        method=ode_method, atol=ATOL)

                for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                    for dst_stn in range(model_data.n_stations):
                        #y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        
                        #traj = np.frombuffer(twrap[src_stn][dst_stn].get_obj())
                        np.frombuffer(twrap[src_stn][dst_stn].get_obj())[:] = x_t.y[traj_cells[cell_idx].get_delay_idx(i, dst_stn), :]

                xiterlock.acquire()
                #x_iter = np.frombuffer(iterwrap.get_obj()).reshape(x_iter_shape)
                np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[traj_cells[cell_idx].get_idx(), :] = x_t.y
                xiterlock.release()

                if cell_limit:
                    if len(x_res) == 0:
                        error_score = float("inf")
                    else:
                        error_score = (abs(x_t.y - x_res[-1][traj_cells[cell_idx].get_idx(),:])).max()

                    if error_score < epsilon:
                        lc_lock.acquire()
                        limited_cells[cell_idx] = iter_no
                        lc_lock.release()
                    #s_ct = traj_cells[cell_idx].s_in_cell
                    if not (x_t.y[-traj_cells[cell_idx].s_in_cell,:] < 1).any():
                        lc_lock.acquire()
                        limited_cells[cell_idx] = iter_no
                        lc_lock.release()
                finished_cells[cell_idx] = True

            
            if last_iter:
                # reverse things on the last iteration, as the limited cells need to be ran again
                if cell_idx in limited_cells and limited_cells[cell_idx] < iter_no - 1:
                    cell_threads[cell_idx] = Process(target=cell_fn, 
                        args=(cell_idx,))#, twrap, iwrap))#, lc_lock, xiterlock, limited_cells))
                else:
                    cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                        args=(cell_idx,))#, twrap, iwrap))#, lc_lock, xiterlock))

            elif cell_limit and cell_idx in limited_cells:
                cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                    args=(cell_idx,))#, twrap, iwrap))#, lc_lock, xiterlock))
            else:
                cell_threads[cell_idx] = Process(target=cell_fn, 
                    args=(cell_idx,))#, twrap, iwrap))#, lc_lock, xiterlock, limited_cells))
        
        # matt - innovation - reduce number of threads to number of cores
        n_cores_avail = 3
        running_cells = set()
        cell_iter = 0
        while cell_iter < model_data.n_cells:
            if len(running_cells) < n_cores_avail:
                cell_threads[cell_iter].start()
                running_cells.add(cell_iter)
                #print(f"created thread for {cell_iter}")
                cell_iter += 1
            else:
                for c in running_cells:
                    if c in finished_cells:
                        cell_threads[c].join()
                        #print(f"removed thread for {c}")
                        running_cells.remove(c)

        while len(running_cells) > 0:
            cell_to_remove = -1
            for c in running_cells:
                if c in finished_cells:
                    cell_threads[c].join()
                    #print(f"removed thread for {c}")
                    cell_to_remove = c
                    break
            if cell_to_remove != -1:
                running_cells.remove(c)

        x_iter = np.frombuffer(iwrap.get_obj()).reshape((n_entries, n_time_points + 1))
        
        x_res.append(x_iter)

        if last_iter:
            toc = time.perf_counter()
            all_res.append([time_points, x_res[-1], toc - tic, n_iterations])
            break


        for srt_stn in range(model_data.n_stations):
            for dst_stn in range(model_data.n_stations):
                trajectories[srt_stn][dst_stn] = np.frombuffer(twrap[srt_stn][dst_stn].get_obj())

        if len(x_res) == 1:
            error_score = float("inf")
        else:
            error_score = (abs(x_res[-1] - x_res[-2])).max()

        h_rec = 0
        if error_score < epsilon:
            if cell_limit:
                # run limited cells one last time
                last_iter = True
            else:
                toc = time.perf_counter()
                all_res.append([time_points, x_res[-1], toc - tic - h_time + h_rec, n_iterations])
                break
        
        print(f"Iteration complete, time: {time.perf_counter()-tic}, error: {error_score}")
        
    print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

    return all_res

def run_discrete(model_data, traj_cells, ode_method, step_size, inflate=True, current_vector="none"):
    print("Running Discrete-Step Submodeling")

    tic = time.perf_counter()
    n_entries = model_data.n_cells**2 + model_data.n_stations
    x_arr = np.array([])
    delay_phase_ct = [sum([len(x) for x in model_data.mu[i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    
    trajectories = [[0 for j in range(traj_cells[end_cell].in_offset)] for end_cell in range(model_data.n_cells)]

    if current_vector == "none":
        current_vector = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
                for cell_idx in range(model_data.n_cells)
        ]
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
                for phase, phase_rate in enumerate(model_data.mu[cell_idx][next_cell]):
                    phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                    
                    last_val = float(x_t.y[phase_qty_idx, -1])
                    current_vector[cell_idx][phase_qty_idx] = last_val
                    new_trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase] = last_val

        if inflate:
            end_bikes = sum([sum(x) for x in new_vector])
            # do loss by number of bikes in delays
            loss_ptg = float(n_bikes-end_bikes)/n_bikes
            inflation_factor = 1/(1-loss_ptg)
            for next_cell in range(model_data.n_cells):
                for phase, phase_rate in enumerate(model_data.mu[cell_idx][next_cell]):
                    new_trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase] *= inflation_factor
            for cell_idx in range(model_data.n_cells):
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
        self.time_end = 1.0
        self.steps_per_dt = 100
        self.const_bike_load = 6
        self.max_iterations = 100

def run_hour(hour, first_vec="none", first_vec_sim="none"):
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
    run_iteration(model_data, traj_cells, "RK45", 0.2, current_vector=first_vec) # HOOK
    time_points, x_arr, time_val, last_vector_sim = run_simulation(model_data,6, current_vector=first_vec_sim)
    time_points, x_arr, time_val, last_vector = run_discrete(model_data, traj_cells, "RK45", 0.2, current_vector=first_vec) # HOOK
 
    station_res = [0 for i in range(n_stations)]
    station_simres = [0 for i in range(n_stations)]

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
    
    return [last_vector, last_vector_sim]
    



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
