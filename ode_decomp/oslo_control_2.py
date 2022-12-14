import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import sklearn
import math
import re
import os
import os.path
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
import sys

import sklearn.cluster

import spatial_decomp_station
import spatial_decomp_strict

import comp
from oslo_data import *
import large_fluid
import get_oslo_data

import pickle

from multiprocessing import Process, Lock, Manager
import multiprocessing.shared_memory

import gc

import ctypes

data_folder = "oslo_data_3_big"
out_folder = "oslo_out"

# Parameters
TIME_POINTS_PER_HOUR = 100
ATOL = 10**(-5)
RATE_MULTIPLIER = 1
SOLVER = "RK45"
TEST_PARAM = False
REQUIRE_POS_GRAD = False
CAPACITY = 15
PRICE_X_THRESHOLD = 0.85


MAX_PRICE = 1.5
MIN_PRICE = 0.5

MAX_PRICE_STEP = 0.3

#random.seed(300)

# how to control the price weighing within a cell
# "proportionate": increases go first to the least loaded station
# "equal": same price across all stations within cell
# "inbound": price determines demand into a cell rather than out.
PRICE_IN_CELL = "equal" 


def run_control(model_data, traj_cells, ode_method, epsilon, cell_limit=False, cell_inc="none", current_vector="none", trajectories="none", prior_res="none", time_length="default"):
    print(f"Running Trajectory-Based Control (parallelized, limit: {cell_limit})")
    all_res = []

    if time_length == "default":
        time_length = model_data.time_end

    tic = time.perf_counter()

    n_entries = model_data.n_stations**2 + model_data.n_stations
    x_res = []

    h_cells = {}
    
    limited_cells = Manager().dict()
    included_cells = Manager().dict()
    cell_include = cell_inc != "none"

    if cell_include:
        for k in cell_inc:
            included_cells[k] = 1


    h_time  = 0

    total_reward = 0

    delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i

    profits = multiprocessing.Array(ctypes.c_double, int(model_data.n_cells))
    arrivals = multiprocessing.Array(ctypes.c_double, int(model_data.n_cells))
    bounces = multiprocessing.Array(ctypes.c_double, int(model_data.n_cells))
    
    if current_vector == "none":
        current_vector = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
                for cell_idx in range(model_data.n_cells)
        ]
    
    x_idx_cell = []
    cur_x_idx = 0
    for traj_cell in traj_cells:
        x_idx_cell.append(cur_x_idx)
        cur_x_idx += traj_cell.x_size()
    x_idx_cell.append(cur_x_idx)

    n_time_points = int(time_length*TIME_POINTS_PER_HOUR)
    time_points = [(i*time_length)/n_time_points for i in range(n_time_points+1)]
    
    if trajectories == "none":
        #trajectories = [[np.zeros(n_time_points+1) for start_cell in range(traj_cells[end_cell].in_offset)] for end_cell in range(model_data.n_cells)]
        trajectories = [[np.zeros(n_time_points+1) for start_cell in range(model_data.n_cells*2)] for end_cell in range(model_data.n_cells)]
    
    if prior_res != "none":
        x_res = [prior_res]
    
    for traj_cell in traj_cells:
        traj_cell.set_timestep(time_length/n_time_points)

    station_vals = []

    n_iterations = 0

    non_h_idx = []

    last_iter = False



    def limited_cell_fn(cell_idx, trajwrap, iterwrap, profitwrap):
        """istart = x_idx_cell[cell_idx]
        iend = istart + traj_cells[cell_idx].x_size()
        if len(x_res) != 0:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = x_res[-1][istart:iend, :]
        else:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = 0
                
        for next_cell in range(model_data.n_cells):
            for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase]
        """
        finished_cells[cell_idx] = True

    def cell_fn(cell_idx, trajwrap, iterwrap, profitwrap):
        traj_cells[cell_idx].set_trajectories(trajectories[cell_idx])
        
        istart = x_idx_cell[cell_idx]
        iend = istart + traj_cells[cell_idx].x_size()

        x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array_2phase, [0, time_length], list(current_vector[cell_idx]) + [0,0,0,0],
                                t_eval=time_points, 
                                method=ode_method, atol=ATOL)
                
        for next_cell in range(model_data.n_cells):
            for phase_idx in range(2):
            #for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                phase = len(model_data.mu[0][cell_idx][next_cell]) - 2 + phase_idx
                phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                    
                #np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = x_t.y[phase_qty_idx, :]
                    
                #np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = x_t.y[phase_qty_idx, :]
                    
                np.frombuffer(twrap[next_cell][cell_idx*2 + phase_idx].get_obj())[:] = x_t.y[phase_qty_idx, :]

        xiterlock.acquire()
        np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = x_t.y[:-4,:]
        xiterlock.release()

        profitlock.acquire()
        profitwrap.get_obj()[0] += x_t.y[-1,-1] # profit
        profitwrap.get_obj()[1] += x_t.y[-2,-1] # regret
        arrivals.get_obj()[cell_idx] = x_t.y[-3,-1] # arrivals
        bounces.get_obj()[cell_idx] = x_t.y[-4,-1] # bounces
        profits.get_obj()[cell_idx] = x_t.y[-1,-1]
        profitlock.release()

        if cell_limit:
            if len(x_res) == 0:
                error_score = float("inf")
            else:
                error_score = (abs(x_t.y[:-4,:] - x_res[-1][traj_cells[cell_idx].get_idx(),:])).max()

            if error_score < epsilon:
                lc_lock.acquire()
                limited_cells[cell_idx] = iter_no
                lc_lock.release()
                
            if not (x_t.y[-traj_cells[cell_idx].s_in_cell,:] < 1).any():
                lc_lock.acquire()
                limited_cells[cell_idx] = iter_no
                lc_lock.release()
        elif cell_include:
            if len(x_res) == 0:
                error_score = float("inf")
            else:
                error_score = (abs(x_t.y[:-4,:] - x_res[-1][istart:iend,:])).max()

            if error_score < epsilon:
                lc_lock.acquire()
                del included_cells[cell_idx]
                lc_lock.release()
            else:
                for next_cell in range(model_data.n_cells):
                    if next_cell == cell_idx or next_cell in included_cells:
                        continue

                    #for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                    for phase_idx in range(2):
                            
                        #if abs(np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] - trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase]).max() >= epsilon:   
                        if abs(np.frombuffer(twrap[next_cell][cell_idx*2 + phase_idx].get_obj())[:] - trajectories[next_cell][cell_idx*2 + phase_idx]).max() >= epsilon:
                            print("adding in cell...")
                            included_cells[next_cell] = 1
                            added = True
                            break


        finished_cells[cell_idx] = True
    


    for iter_no in range(model_data.max_iterations):
        n_iterations = iter_no + 1

        gc.collect()

        iwrap = multiprocessing.Array(ctypes.c_double, int(n_entries*(n_time_points+1)))
        #twrap = [[multiprocessing.Array(ctypes.c_double, n_time_points+1) for j in range(traj_cells[i].in_offset)] for i in range(model_data.n_cells)]
        # change trajectories to only go to 2 phases
        twrap = [[multiprocessing.Array(ctypes.c_double, n_time_points+1) for j in range(model_data.n_cells*2)] for i in range(model_data.n_cells)]
        pwrap = multiprocessing.Array(ctypes.c_double, 2) 



        x_iter_shape = (n_entries, n_time_points + 1)

        # <HOOK>
        if len(x_res) != 0:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[:,:] = x_res[-1][:, :]
        else:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[:, :] = 0

        for cell_idx in range(model_data.n_cells):
            for next_cell in range(model_data.n_cells):
                #for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                for phase_idx in range(2):
                    #np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase]
                    np.frombuffer(twrap[next_cell][cell_idx*2 + phase_idx].get_obj())[:] = trajectories[next_cell][cell_idx*2 + phase_idx]
        # </HOOK>


        cell_threads = [None for i in range(model_data.n_cells)]
        x_t_cell = [None for i in range(model_data.n_cells)]

        finished_cells = Manager().dict()

        for cell_idx in range(model_data.n_cells):
            lc_lock = Lock()
            xiterlock = Lock()
            profitlock = Lock()
            
            if last_iter:
                # reverse things on the last iteration, as the limited cells need to be ran again
                if cell_idx in limited_cells and limited_cells[cell_idx] < iter_no - 1:
                    finished_cells[cell_idx] = False
                    cell_threads[cell_idx] = Process(target=cell_fn, 
                        args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock, limited_cells))
                else:
                    finished_cells[cell_idx] = True
                    #cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                    #    args=(cell_idx, twrap, iwrap, prwap))#, lc_lock, xiterlock))

            elif cell_limit and cell_idx in limited_cells:
                finished_cells[cell_idx] = True
                #cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                #    args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock))
            elif cell_include and cell_idx not in included_cells:
                finished_cells[cell_idx] = True
                #cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                #    args=(cell_idx, twrap, iwrap, pwrap))
            else:
                finished_cells[cell_idx] = False
                cell_threads[cell_idx] = Process(target=cell_fn, 
                    args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock, limited_cells))
        
        # matt - innovation - reduce number of threads to number of cores
        n_cores_avail = 3 if ode_method == "BDF" else 10
        running_cells = set()
        cell_iter = 0
        while cell_iter < model_data.n_cells:
            if len(running_cells) < n_cores_avail:
                if not finished_cells[cell_iter]:
                    cell_threads[cell_iter].start()
                    running_cells.add(cell_iter)
                    #print(f"created thread for {cell_iter}")
                cell_iter += 1
                
            else:
                cell_to_remove = -1
                for c in running_cells:
                    if c in finished_cells:
                        cell_threads[c].join()
                        #print(f"removed thread for {c}")
                        cell_to_remove = c
                        break
                if cell_to_remove != -1:
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

        print(f"Reward: {pwrap.get_obj()[0]}, regret: {pwrap.get_obj()[1]}, arrivals: {sum(arrivals.get_obj())}, bounces: {sum(bounces.get_obj())}")

        total_reward = pwrap.get_obj()[0]
        regret = pwrap.get_obj()[1]
        arrival_list = list(arrivals.get_obj())
        bounce_list = list(bounces.get_obj())


        for end_cell in range(model_data.n_cells):
            for traj_start in range(len(trajectories[end_cell])):
                trajectories[end_cell][traj_start] = np.frombuffer(twrap[end_cell][traj_start].get_obj()).copy()


        if last_iter:
            toc = time.perf_counter()
            all_res.append([time_points, x_res[-1], toc - tic, n_iterations])
            break
        elif len(x_res) == 1:
            error_score = float("inf")
        elif cell_include and len(included_cells.keys()) == 1:
            error_score = 0
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
        
    out_vector = []

    for cell_idx, traj_cell in enumerate(traj_cells):
        out_vector.append(x_res[-1][x_idx_cell[cell_idx]:x_idx_cell[cell_idx+1], -1])

    profits = list(np.frombuffer(profits.get_obj()))

    
    print(f"Trajectory-Based Iteration finished, time: {toc-tic}, profit {sum(profits)}")
    return [all_res, x_res[-1], trajectories, out_vector, total_reward, profits, regret, arrival_list, bounce_list]


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
        self.const_bike_load = 10
        self.max_iterations = 100

def run_control_period_optuna(start_hour, end_hour, prices, cell_levels, prior_cell_levels, 
            final_cell_levels,
            finite_difference_x = 0.1, finite_difference_price=0.1, run_xdiff=True, run_price=True, cache=True, 
            bounce_cost=0.0, rebalancing_cost=0.0,
            annealing_steps=100, starting_temperature=1.0, annealing_alpha=0.95, change_one=True):
    print(f"running hours: {start_hour} -> {end_hour}")

    # demands
    in_demands = []
    in_probabilities = []
    out_demands = []
    station_iterres = [0 for i in range(n_stations)]
    station_sampres = [0 for i in range(n_stations)]

    mu = []
    phi = []

    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
        hr = hr % 24
        in_demands_hr, in_probabilities_hr, out_demands_hr = get_oslo_data.get_demands(hr, cell_to_station, station_to_cell)

        # phase processes
        mu_hr, phi_hr = get_oslo_data.get_cox_data(hr, len(cell_to_station))

        in_demands.append(in_demands_hr)
        in_probabilities.append(in_probabilities_hr)
        out_demands.append(out_demands_hr)
        
        mu.append(mu_hr)
        phi.append(phi_hr)
        
    starting_bps = get_oslo_data.get_starting_bps()
    model_data = ModelData(n_stations, n_cells, starting_bps, mu, phi, in_demands, in_probabilities, out_demands)

    # set first_vec_iter based on cell_levels
    delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    first_vec_iter = [
        [0.0 for i in range(delay_phase_ct[cell_idx])] +
        cell_levels[cell_idx]
            for cell_idx in range(model_data.n_cells)
    ]
        
    # build StrictTrajCellCox for each station cell
    # go from [hr][cell_idx] -> [cell_idx][hr]
    rev_hr_cell = lambda x, cell_idx: [xhr[cell_idx] for xhr in x]
    traj_cells = [
        spatial_decomp_strict.StrictTrajCellCoxControl(cell_idx, 
            cell_to_station[cell_idx], 
            mu, phi, 
            rev_hr_cell(in_demands,cell_idx), 
            rev_hr_cell(in_probabilities,cell_idx), 
            rev_hr_cell(out_demands,cell_idx),
            capacities[cell_idx])

            for cell_idx in range(model_data.n_cells)
    ]
    
    for i, traj_cell in enumerate(traj_cells):
        if PRICE_IN_CELL == "proportionate":
            traj_cell.set_price_proportionate(prices[i], first_vec_iter[i], MIN_PRICE, MAX_PRICE)
        elif PRICE_IN_CELL == "inbound":
            traj_cell.set_inbound_prices(prices)
            traj_cell.use_inbound_price = True
        else:
            traj_cell.set_price(prices[i])

    # run initial model
    def objective(trial):
        ares, lastres, trajectories, last_vector_iter, total_reward, profits, regret, arrivals, bounces = run_control(model_data, traj_cells, SOLVER, 0.5, current_vector=first_vec_iter, time_length=float(end_hour-start_hour))
        gc.collect()
        profits = [x - (bounce_cost*y) for x,y in zip(profits, bounces)]

        reb_cost = 0

        for cell_idx in range(n_cells):
            for stn_idx in range(len(list(cell_to_station[cell_idx]))):
                final_level = last_vector_iter[cell_idx][-stn_idx]
                if final_cell_levels == "same":
                    next_level = cell_levels[cell_idx][-stn_idx]
                else:
                    next_level = final_cell_levels[cell_idx][-stn_idx]
                reb_cost += rebalancing_cost*abs(final_level - next_level)

        print(f"reward: {total_reward}, bounces: {sum(bounces)}, profits: {sum(profits)}, rebalancing: {reb_cost}")
        total_reward -= bounce_cost*(sum(bounces))

def run_control_period_sa(start_hour, end_hour, prices, cell_levels, prior_cell_levels, 
            final_cell_levels,
            finite_difference_x = 0.1, finite_difference_price=0.1, run_xdiff=True, run_price=True, cache=True, 
            bounce_cost=0.0, rebalancing_cost=0.0,
            annealing_steps=100, starting_temperature=1.0, annealing_alpha=0.95, change_one=True):
    print(f"running hours: {start_hour} -> {end_hour}")

    # demands
    in_demands = []
    in_probabilities = []
    out_demands = []
    station_iterres = [0 for i in range(n_stations)]
    station_sampres = [0 for i in range(n_stations)]

    mu = []
    phi = []

    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
        hr = hr % 24
        in_demands_hr, in_probabilities_hr, out_demands_hr = get_oslo_data.get_demands(hr, cell_to_station, station_to_cell)

        # phase processes
        mu_hr, phi_hr = get_oslo_data.get_cox_data(hr, len(cell_to_station))

        in_demands.append(in_demands_hr)
        in_probabilities.append(in_probabilities_hr)
        out_demands.append(out_demands_hr)
        
        mu.append(mu_hr)
        phi.append(phi_hr)
        
    starting_bps = get_oslo_data.get_starting_bps()
    model_data = ModelData(n_stations, n_cells, starting_bps, mu, phi, in_demands, in_probabilities, out_demands)

    # set first_vec_iter based on cell_levels
    delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    first_vec_iter = [
        [0.0 for i in range(delay_phase_ct[cell_idx])] +
        cell_levels[cell_idx]
            for cell_idx in range(model_data.n_cells)
    ]
        
    # build StrictTrajCellCox for each station cell
    # go from [hr][cell_idx] -> [cell_idx][hr]
    rev_hr_cell = lambda x, cell_idx: [xhr[cell_idx] for xhr in x]
    traj_cells = [
        spatial_decomp_strict.StrictTrajCellCoxControl(cell_idx, 
            cell_to_station[cell_idx], 
            mu, phi, 
            rev_hr_cell(in_demands,cell_idx), 
            rev_hr_cell(in_probabilities,cell_idx), 
            rev_hr_cell(out_demands,cell_idx),
            capacities[cell_idx])

            for cell_idx in range(model_data.n_cells)
    ]
    
    for i, traj_cell in enumerate(traj_cells):
        if PRICE_IN_CELL == "proportionate":
            traj_cell.set_price_proportionate(prices[i], first_vec_iter[i], MIN_PRICE, MAX_PRICE)
        elif PRICE_IN_CELL == "inbound":
            traj_cell.set_inbound_prices(prices)
            traj_cell.use_inbound_price = True
        else:
            traj_cell.set_price(prices[i])

    # run initial model
    filename = "control_res/res_{}_{}".format(start_hour,end_hour)

    if os.path.isfile(filename) and cache:
        with open(filename, "rb") as f:
            ares, lastres, trajectories, last_vector_iter, trajectories, total_reward, profits, reb_cost = pickle.load(f)
    else:
        # all_res, x_res[-1], trajectories, out_vector, total_reward, profits, regret, arrivals
        ares, lastres, trajectories, last_vector_iter, total_reward, profits, regret, arrivals, bounces = run_control(model_data, traj_cells, SOLVER, 0.5, current_vector=first_vec_iter, time_length=float(end_hour-start_hour))
        gc.collect()
        profits = [x - (bounce_cost*y) for x,y in zip(profits, bounces)]

        reb_cost = 0

        for cell_idx in range(n_cells):
            for stn_idx in range(len(list(cell_to_station[cell_idx]))):
                final_level = last_vector_iter[cell_idx][-stn_idx]
                if final_cell_levels == "same":
                    next_level = cell_levels[cell_idx][-stn_idx]
                else:
                    next_level = final_cell_levels[cell_idx][-stn_idx]
                reb_cost += rebalancing_cost*abs(final_level - next_level)

        print(f"reward: {total_reward}, bounces: {sum(bounces)}, profits: {sum(profits)}, rebalancing: {reb_cost}")
        total_reward -= bounce_cost*(sum(bounces))

        if cache:
            with open(filename, "xb") as f:
                pickle.dump([ares, lastres, trajectories, last_vector_iter, trajectories, total_reward, profits, regret, reb_cost],f)

    overall_delta = 0
    cell_idxes = [cell_idx for cell_idx in range(n_cells)]
    random.shuffle(cell_idxes)
    for cell_n, cell_idx in enumerate(cell_idxes):
        print("--------------------------------------------------------------------------------------")
        print(f"annealing: {cell_idx} (n: {cell_n})")
        cell_temp = starting_temperature

        acache = {}

        for annealing_iteration in range(annealing_steps):
            change_x = False
            change_price = False
            if run_xdiff and run_price:
                if random.random() < PRICE_X_THRESHOLD:
                    change_x = True
                else:
                    change_price = True
            elif run_xdiff:
                change_x = True
            elif run_price:
                change_price = True


            if change_x:
                x0_total = sum(first_vec_iter[cell_idx])
                station_idx = random.randrange(len(cell_to_station[cell_idx]))
                change = random.choice([-1,1])
                
                if change_one:
                    if first_vec_iter[cell_idx][-station_idx] == 0:
                        # prevent negative starting values
                        change = 1
                    first_vec_iter[cell_idx][-station_idx] += change
                else:
                    if first_vec_iter[cell_idx][-station_idx] == 0:
                        # prevent negative starting values
                        change = 1
                    for station_idx in range(len(cell_to_station[cell_idx])):
                        first_vec_iter[cell_idx][-station_idx] += change
                if change_one or (first_vec_iter[cell_idx][-1], traj_cells[cell_idx].price) not in acache:
                    ares, lastres, sample_trajectories, sample_last_vector, new_total_reward, new_profits, new_regret, new_arrivals,new_bounces = run_control(model_data, 
                        traj_cells, SOLVER, 0.02, trajectories=trajectories, 
                        prior_res=lastres, cell_inc=[cell_idx], current_vector=first_vec_iter, time_length=float(end_hour-start_hour))
                    gc.collect()

                    new_profits = [x - bounce_cost*y for x, y in zip(new_profits, new_bounces)]

                    local_profit_delta = sum([(x - profits[i]) for i, x in enumerate(new_profits) if new_profits[i] != 0])

                    # update starting & ending rebalancing costs

                    final_rebalancing_cost = 0
                    init_rebalancing_cost = 0

                    if prior_cell_levels != "same":
                        raise Exception("This should take into account the change in bicycle levels in the previous stage")

                    for dst_cell_idx, new_cell_profit in enumerate(new_profits):
                        # use profits to see if the cell has been re-ran
                        if new_cell_profit != 0:
                            s_in_cell = len(cell_to_station[dst_cell_idx])
                            for stn in range(s_in_cell):
                                new_station_level = sample_last_vector[dst_cell_idx][-stn]
                                if final_cell_levels == "same":
                                    init_station_level = first_vec_iter[cell_idx][-stn]
                                else:
                                    init_station_level = final_cell_levels[cell_idx][-stn]
                                orig_station_level = last_vector_iter[dst_cell_idx][-stn]
                                final_rebalancing_cost += 0.5 * rebalancing_cost * (abs(new_station_level - init_station_level) - abs(orig_station_level-init_station_level))

                    cell_delta = local_profit_delta - (init_rebalancing_cost + final_rebalancing_cost)
                    acache[(first_vec_iter[cell_idx][-1], traj_cells[cell_idx].price)] = cell_delta
                else:
                    cell_delta = acache[(first_vec_iter[cell_idx][-1], traj_cells[cell_idx].price)]

                if cell_delta <= 0 and random.random() > math.exp(cell_delta/cell_temp):
                    if change_one:
                        first_vec_iter[cell_idx][-station_idx] -= change
                    else:
                        for station_idx in range(len(cell_to_station[cell_idx])):
                            first_vec_iter[cell_idx][-station_idx] -= change
                else:
                    s_in_cell = len(cell_to_station[cell_idx])
                    cell_levels[cell_idx] = copy.deepcopy(first_vec_iter[cell_idx][-s_in_cell:])
                    overall_delta += cell_delta
                    trajectories = sample_trajectories
                    gc.collect()

                    for dst_cell_idx, new_cell_profit in enumerate(new_profits):
                        # use profits to see if the cell has been re-ran
                        if new_cell_profit != 0:
                            last_vector_iter[dst_cell_idx] = sample_last_vector[dst_cell_idx]

                
                        
            if change_price:
                direction = random.choice([-1,1])
                print(f"price deriv: {cell_idx}")
                sum_dprofit = 0
                sum_dx = [0 for i in range(n_cells)]
                found_pos = False

                old_prices = copy.deepcopy(traj_cells[cell_idx].prices)
                old_price = traj_cells[cell_idx].price
                new_price = old_price + (direction*finite_difference_price)


                if change_one or (first_vec_iter[cell_idx][-1], new_price) not in acache:
                    cost_of_change = 0
                    if PRICE_IN_CELL == "proportionate":
                        traj_cells[cell_idx].set_price_proportionate((old_price + (direction*finite_difference_price)), first_vec_iter[cell_idx], MIN_PRICE, MAX_PRICE)
                    elif PRICE_IN_CELL == "inbound":
                        cost_of_change += traj_cells[cell_idx].simulate_inbound_change(direction*finite_difference_price)
                    else:
                        traj_cells[cell_idx].set_price(traj_cells[cell_idx].price + direction*finite_difference_price)

                    starting_vec = [(x if i != cell_idx else x) for i, x in enumerate(first_vec_iter)]
                    ares, lastres, sample_trajectories, sample_last_vector, new_total_reward, new_profits, new_regret, new_arrivals, new_bounces = run_control(model_data, traj_cells, SOLVER, 0.02, trajectories=trajectories, 
                                    prior_res=lastres, cell_inc=[cell_idx], current_vector=first_vec_iter, time_length=float(end_hour-start_hour))
                    gc.collect()
                    new_profits = [x - (bounce_cost*y) for x, y in zip(new_profits, new_bounces)]
                    local_profit_delta = sum([(x - profits[i]) for i, x in enumerate(new_profits) if new_profits[i] != 0])
                    
                    # update starting & ending rebalancing costs
                    
                    final_rebalancing_cost = 0
                    for dst_cell_idx, new_cell_profit in enumerate(new_profits):
                        # use profits to see if the cell has been re-ran
                        if new_cell_profit != 0:
                            s_in_cell = len(cell_to_station[dst_cell_idx])
                            for stn in range(s_in_cell):
                                new_station_level = sample_last_vector[dst_cell_idx][-stn]
                                if final_cell_levels == "same":
                                    init_station_level = first_vec_iter[cell_idx][-stn]
                                else:
                                    init_station_level = final_cell_levels[cell_idx][-stn]
                                orig_station_level = last_vector_iter[dst_cell_idx][-stn]
                                final_rebalancing_cost += 0.5 * rebalancing_cost * (abs(new_station_level - init_station_level) - abs(orig_station_level-init_station_level))

                    cell_delta = local_profit_delta - final_rebalancing_cost
                    acache[(first_vec_iter[cell_idx][-1], new_price)] = cell_delta
                else:
                    cell_delta = acache[(first_vec_iter[cell_idx][-1], new_price)]

                if cell_delta <= 0 and random.random() > math.exp(cell_delta/cell_temp):
                    #traj_cells[cell_idx].set_price(traj_cells[cell_idx].price - (direction*finite_difference_price))
                    if PRICE_IN_CELL == "inbound":
                        traj_cells[cell_idx].revert_inbound_change(direction*finite_difference_price)
                    else:
                        traj_cells[cell_idx].price = old_price
                        traj_cells[cell_idx].set_prices_list(old_prices)
                else:
                    prices[cell_idx] = old_price + (direction*finite_difference_price)
                    overall_delta += cell_delta
                    trajectories = sample_trajectories
                    gc.collect()

                    for dst_cell_idx, new_cell_profit in enumerate(new_profits):
                        # use profits to see if the cell has been re-ran
                        if new_cell_profit != 0:
                            last_vector_iter[dst_cell_idx] = sample_last_vector[dst_cell_idx]

            cell_temp *= annealing_alpha
                        
    print(f"simulated annealing finished. expecting profits of {overall_delta}")
    with open(f"{out_folder}/res_iter_control_{start_hour}_{end_hour}", "w") as f:
        json.dump(station_iterres, f)
        
    return [station_iterres, last_vector_iter, prices, cell_levels, total_reward, profits, regret, reb_cost]


# what we need:
#   derivatives:
#       final state vs initial state <- surrogate modeling
#       reward vs initial state      <- surrogate modeling
#       final state vs price         <- estimate via sum of delays
#       reward vs price              <- estimate via prior reward and departure transform


def start_step(alpha, vec_iter, last_vector_iter, dprofit_dx, dxf_dx, rebalancing_cost):
    n_bikes = sum([sum(x) for x in vec_iter])
    # 1. find gradient for rebalancing cost
    #   cost(xf) = sum[rebalancing_cost*max(xf_i-x0_i,0)]
    #   - to differentiate this we use log-sum-exp approximation
    #   => cost(xf) ~= sum[rebalancing_cost*log(exp(xf_i-x0-i)+1)]
    #   => in addition, assume dxf_dx0 is constant within each cell
    #   =>  1. let delta_x = xf - x_0, c = rebalancing_cost
    #       2. dcost/dx0 = c*(exp{delta_x}/exp{delta_x+1})*(dxf/dx0 - 1) <- when looking at an identical x
    #       3. dcost/dx0 = c*(exp{delta_x}/exp{delta_x+1})*(dxf/dx0) <- non-identical x
    #   => sum over each cell

    raise Exception("revsit calculations (carefully..add in weights?)")

    rebalancing_deriv = [] # this is negative in the actual term
    exp_terms = []
    for cell_idx in range(n_cells):
        xstart = vec_iter[cell_idx]
        xend   = last_vector_iter[cell_idx]
        exp_xdelta = [math.exp(xf-x0) for x0, xf in zip(xstart, xend)]
        exp_term = [ex/(ex+1) for ex in exp_xdelta]
        exp_terms.append(exp_term)
    
    for cell_idx in range(n_cells):
        cost_delta = 0
        for other_cell in range(n_cells): 
            indicator = 1 if cell_idx == other_cell else 0
            cost_delta += sum([rebalancing_cost*ext*(dxf_dx[cell_idx][other_cell]-indicator) for ext in exp_terms[other_cell]])
        rebalancing_deriv.append(cost_delta)

    # 2. find overall_gradient w.r.t. starting point
    overall_deriv = [pd-rd for rd,pd in zip(rebalancing_deriv, dprofit_dx)]

    # 3. find new vec_iter
    new_vec_iter = []
    for cell_idx in range(n_cells):
        cell_vec = [x + alpha*(overall_deriv[cell_idx]) for x in vec_iter[cell_idx]]
        new_vec_iter.append(cell_vec)


    # 4. update vec_iter to account for constraints
    # current method, clamp then scale (e.g. make sure all values are non-negative and then multiply by a scaling factor)
    # clamping step..
    new_n_bikes = 0
    for cell_idx in range(n_cells):
        # this below is a bit complex but it forces all delay terms to be equal to 0
        stn_start = len(new_vec_iter[cell_idx])-len(cell_to_station[cell_idx])
        new_vec_iter[cell_idx] = [max(x, 0) if i >= stn_start else 0 for i,x in enumerate(new_vec_iter[cell_idx])]
        new_n_bikes += sum(new_vec_iter[cell_idx])
    
    inf_factor = n_bikes/new_n_bikes
    
    for cell_idx in range(n_cells):
        new_vec_iter[cell_idx] = [x*inf_factor for x in new_vec_iter[cell_idx]]
    
    # 5. calculate rebalancing cost for the previous iteration
    reb_costs = []
    for cell_idx in range(n_cells):
        xstart = vec_iter[cell_idx]
        xend   = last_vector_iter[cell_idx]
        reb_costs.append(sum([rebalancing_cost*max(xf-x0,0) for x0, xf in zip(xstart, xend)]))
    
    return new_vec_iter, sum(reb_costs)


def get_delay_phase_ct(start_hour, end_hour):
    # code duplication wrt get_oslo_data
    mu = []
    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
        # phase processes
        mu_hr, phi_hr = get_oslo_data.get_cox_data(hr, len(cell_to_station))
        
        mu.append(mu_hr)
    
    delay_phase_ct = [sum([len(x) for x in mu[0][i]]) for i in range(n_cells)] # number of phases in the process that starts at i
    
    return delay_phase_ct


def get_first_vec_iter(start_hour, end_hour):
    # code duplication wrt get_oslo_data
    mu = []
    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
        # phase processes
        mu_hr, phi_hr = get_oslo_data.get_cox_data(hr, len(cell_to_station))
        
        mu.append(mu_hr)
    
    delay_phase_ct = [sum([len(x) for x in mu[0][i]]) for i in range(n_cells)] # number of phases in the process that starts at i

    if TEST_PARAM:
        first_vec_iter = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            #[6.0 for x in list(cell_to_station[cell_idx])]
            [random.random()*3.0 for x in list(cell_to_station[cell_idx])]
                for cell_idx in range(n_cells)
        ]
    else:
        first_vec_iter = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [10.0 for x in list(cell_to_station[cell_idx])]
            #[random.random()*12.0 for x in list(cell_to_station[cell_idx])]
                for cell_idx in range(n_cells)
        ]
    return first_vec_iter

def get_rebalancing_deriv_dx(alpha, starting_vec, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, rebalancing_cost, prices, start_hour, end_hour):
    # rebalancing derivative with respect to xf...
    SOFTMAX_PARAM = 1

    avg_probs = get_oslo_data.get_average_probs(start_hour, end_hour, cell_to_station, station_to_cell)

    rebalancing_deriv = [] # this is negative in the actual term
    softmax_deriv = []
    diff_terms = []
    cell_diff_terms = []
    exp_terms = []
    
    for cell_idx in range(n_cells):
        xstart = starting_vec[cell_idx]
        xend   = last_vector_iter[cell_idx]
        total_xf = sum(xend)
        weights  = [xf/total_xf for xf in xend]
        s_in_cell = len(list(cell_to_station[cell_idx]))
        x_delta = [xf - x0 for x0, xf in zip(xstart,xend)][-s_in_cell:]
        cell_diff_terms.append(sum(x_delta))
        diff_terms.append([0.5 if xd > 0 else -0.5 for xd in x_delta])
        exp_diff = [math.exp(SOFTMAX_PARAM*xd) for xd in x_delta] # (c*exp(deltaX))/(deltaX+1) <- approximately dcost/dx_f
        exp_terms.append([((exd)/(exd+1)) for w, exd in zip(weights,exp_diff)]) # then apply it by weight to analyze the impact of the total on the price
        

    for cell_idx in range(n_cells):
        #cost_delta = sum(diff_terms[cell_idx])*(-rebalancing_cost)
        #cost_delta_sm = sum(exp_terms[cell_idx])*(-rebalancing_cost)
        cost_delta = (0.5 if cell_diff_terms[cell_idx] > 0 else -0.5)*rebalancing_cost
        smax_term = math.exp(SOFTMAX_PARAM*cell_diff_terms[cell_idx])
        smin_term = -math.exp(0-(SOFTMAX_PARAM*cell_diff_terms[cell_idx]))
        cost_delta_sm = (0.5*(smax_term/(smax_term+1)) + 0.5*(smin_term/(smin_term+1)))*rebalancing_cost
        #for other_cell in range(n_cells): 
        #    cost_delta += sum([-rebalancing_cost*deltax*avg_probs[other_cell][cell_idx][i] for i,deltax in enumerate(diff_terms[other_cell])])
        #    cost_delta_sm += sum([-rebalancing_cost*exp*avg_probs[other_cell][cell_idx][i] for i,exp in enumerate(exp_terms[other_cell])])
        rebalancing_deriv.append(cost_delta)
        softmax_deriv.append(cost_delta_sm)
    print(f"rebalancing cost: {rebalancing_cost}")
    print(f"derivs: {rebalancing_deriv}")
    print(f"diff terms: {cell_diff_terms}")
    # 5. calculate rebalancing cost for the previous iteration
    reb_costs = []
    estimated_reb_costs = []
    for cell_idx in range(n_cells):
        xstart = starting_vec[cell_idx]
        xend   = last_vector_iter[cell_idx]
        reb_costs.append(sum([rebalancing_cost*max(xf-x0,0) for x0, xf in zip(xstart, xend)]))
        estimated_reb_costs.append(sum([rebalancing_cost*(math.log(math.exp(xf-x0) + 1)) for x0, xf in zip(xstart, xend)]))
    
    return [rebalancing_deriv, sum(estimated_reb_costs)]

def price_step_final(xf_profit, alpha, vec_iter, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, rebalancing_cost, prices, start_hour, end_hour):
    """Adapted from start_step for prices. This assumes that we're in the last stage. 

        There are, of course, slight changes in the calculation. Should be careful here...
    """

    print(f"Original prices: {prices}, alpha: {alpha}")

    # 2. find overall_gradient w.r.t. starting point
    overall_deriv = [pdr+xfp for xfp,pdr in zip(xf_profit, dprofit_dp)]

    print(f"Overall derivatives: {overall_deriv}")
    print(f"Rebalancing derivatives: {xf_profit}")
    print(f"Pricing derivatives: {dprofit_dp}")

    # 3. find new prices
    new_prices = []
    for cell_idx in range(n_cells):
        price_step = (alpha*overall_deriv[cell_idx])
        if price_step > MAX_PRICE_STEP:
            price_step = MAX_PRICE_STEP
        if price_step < -MAX_PRICE_STEP:
            price_step = -MAX_PRICE_STEP
        new_price = prices[cell_idx] + price_step
        new_prices.append(max(min(new_price, MAX_PRICE),MIN_PRICE))
    print(f"New prices: {new_prices}")
        
    
    return new_prices

def optimize_start_optuna(rebalancing_cost, bounce_cost, run_price=True, run_xdiff=True):
    # misnomer, this is actually the overall simulated annealing optimization function
    start_hours = [5]
    hour_delta = 16

    annealing_steps = 10
    temperature = 20
    annealing_alpha = 0.9

    #temperature = temperature * (annealing_alpha**(80))


    iteration = 0

    starting_level = 10.0
    prices = [[1.0 for cell_idx in range(n_cells)] for hr in start_hours]
    cell_levels = [[[starting_level for stn_idx in range(len(list(cell_to_station[cell_idx])))] for cell_idx in range(n_cells)] for hr in start_hours]
    ending_cell_levels = copy.deepcopy(cell_levels)

    iter_profits   = []
    iter_reb_costs = []
    iter_regret    = []

    delay_phase_ct = get_delay_phase_ct(start_hours[0], start_hours[-1] + hour_delta)

    while True:
        filename = "price_res/res_{}".format(iteration)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                vec_iter, last_vector_iter, profit, new_prices, prices, reb_cost = pickle.load(f)
            
            iter_profits.append(profit)
            iter_reb_costs.append(reb_cost)
                
            prices = new_prices
            alpha_p = ((alpha_p-min_alpha_p)*alpha_decay) + min_alpha_p

            iteration += 1

            print(f"previous iterations: {iter_profits}")
            print(f".with rebalancing costs: {iter_reb_costs}")
            print(f"and regret: {iter_regret}")
            continue

        total_profit = 0
        total_regret = 0

        for hour_idx, start_hour in enumerate(start_hours):
            end_hour = start_hour + hour_delta
            n_hours = len(start_hours)
            # in: start_hour, end_hour, first_vec_iter, prices, cell_levels, prior_cell_levels, final_cell_levels,
            res, last_vector_iter, new_prices, new_cell_levels, profit, profits, regret, reb_cost = run_control_period_sa(start_hour,end_hour,
                        copy.deepcopy(prices[hour_idx]), copy.deepcopy(cell_levels[hour_idx]),
                        "same" if (n_hours == 1) else ending_cell_levels[(hour_idx + n_hours - 1) % n_hours], # cell levels before rebalancing
                        "same" if (n_hours == 1) else cell_levels[(hours_idx + 1) % n_hours ], # next cell idx 
                        cache=False, 
                        finite_difference_x=1, 
                        finite_difference_price=0.1,
                        run_price=run_price, 
                        run_xdiff=run_xdiff,
                        bounce_cost=bounce_cost, rebalancing_cost=rebalancing_cost,
                        starting_temperature=temperature,
                        annealing_steps=annealing_steps,
                        annealing_alpha=annealing_alpha,
                        change_one=(iteration > 10)) # start changing levels one-at-a-time after iteration 10
            print(f"original cell levels: {[x[-1] for x in cell_levels[hour_idx]]}")
            print(f"original prices: {prices[hour_idx]}")

            total_profit += profit
            total_regret += regret
            prices[hour_idx] = new_prices
            cell_levels[hour_idx] = new_cell_levels
            temperature = temperature*(annealing_alpha**annealing_steps)

            print(f"new cell levels: {[x[-1] for x in cell_levels[hour_idx]]}")
            print(f"new prices: {prices[hour_idx]}")

            for cell_idx in range(n_cells):
                s_in_cell = len(cell_to_station[cell_idx])
                ending_cell_levels[hour_idx][cell_idx] = last_vector_iter[cell_idx][-s_in_cell:]
        
        iter_profits.append(total_profit)
        iter_reb_costs.append(reb_cost)
        iter_regret.append(total_regret)

        print(f"At iteration {iteration}...profit: {total_profit}, regret: {total_regret}, rebalancing costs: {reb_cost}")

        print(f"previous iterations: {iter_profits}")
        print(f".with rebalancing costs: {iter_reb_costs}")
        print(f"and regret: {iter_regret}")
        
        iteration += 1
     
        
def optimize_start(rebalancing_cost, bounce_cost, run_price=True, run_xdiff=True):
    # misnomer, this is actually the overall simulated annealing optimization function
    start_hours = [5]
    hour_delta = 16

    annealing_steps = 10
    temperature = 20
    annealing_alpha = 0.9

    #temperature = temperature * (annealing_alpha**(20))


    iteration = 0

    starting_level = 10.0
    prices = [[1.0 for cell_idx in range(n_cells)] for hr in start_hours]
    #prices = [[1.1000000000000003, 1.0, 1.2000000000000002, 0.5000000000000001, 0.6000000000000001, 1.2000000000000002, 0.8000000000000003, 1.3000000000000003, 0.9, 1.4000000000000004, 1.0000000000000002, 0.7000000000000001, 1.0, 0.6000000000000001, 1.0, 1.0, 1.1, 1.3, 0.7000000000000003, 1.2000000000000002, 1.2000000000000004, 1.3, 0.30000000000000016, 1.4000000000000004, 1.1, 1.3000000000000003, 0.9, 1.2000000000000002, 0.9999999999999992, 1.1, 1.0, 1.3000000000000003, 0.5999999999999994, 1.1000000000000005, 1.1000000000000003, 0.8, 1.0, 0.7000000000000001, 1.3000000000000003, 0.7999999999999999, 1.1, 1.3000000000000003, 1.4000000000000004, 1.3000000000000003, 1.5000000000000004, 0.6000000000000001, 1.2000000000000002, 1.1000000000000003, 0.5000000000000001, 1.1, 0.9999999999999994, 0.4, 0.40000000000000013, 1.3000000000000003, 0.9, 1.2, 0.6000000000000001, 0.7000000000000001] for hr in start_hours]
    cell_levels = [[[starting_level for stn_idx in range(len(list(cell_to_station[cell_idx])))] for cell_idx in range(n_cells)] for hr in start_hours]
    ending_cell_levels = copy.deepcopy(cell_levels)

    iter_profits   = []
    iter_reb_costs = []
    iter_regret    = []

    delay_phase_ct = get_delay_phase_ct(start_hours[0], start_hours[-1] + hour_delta)

    while True:
        filename = "price_res/res_{}".format(iteration)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                vec_iter, last_vector_iter, profit, new_prices, prices, reb_cost = pickle.load(f)
            
            iter_profits.append(profit)
            iter_reb_costs.append(reb_cost)
                
            prices = new_prices
            alpha_p = ((alpha_p-min_alpha_p)*alpha_decay) + min_alpha_p

            iteration += 1

            print(f"previous iterations: {iter_profits}")
            print(f".with rebalancing costs: {iter_reb_costs}")
            print(f"and regret: {iter_regret}")
            continue

        total_profit = 0
        total_regret = 0

        for hour_idx, start_hour in enumerate(start_hours):
            end_hour = start_hour + hour_delta
            n_hours = len(start_hours)
            # in: start_hour, end_hour, first_vec_iter, prices, cell_levels, prior_cell_levels, final_cell_levels,
            res, last_vector_iter, new_prices, new_cell_levels, profit, profits, regret, reb_cost = run_control_period_sa(start_hour,end_hour,
                        copy.deepcopy(prices[hour_idx]), copy.deepcopy(cell_levels[hour_idx]),
                        "same" if (n_hours == 1) else ending_cell_levels[(hour_idx + n_hours - 1) % n_hours], # cell levels before rebalancing
                        "same" if (n_hours == 1) else cell_levels[(hours_idx + 1) % n_hours ], # next cell idx 
                        cache=False, 
                        finite_difference_x=1, 
                        finite_difference_price=0.1,
                        run_price=run_price, 
                        run_xdiff=run_xdiff,
                        bounce_cost=bounce_cost, rebalancing_cost=rebalancing_cost,
                        starting_temperature=temperature,
                        annealing_steps=annealing_steps,
                        annealing_alpha=annealing_alpha,
                        change_one=(iteration > 10)) # start changing levels one-at-a-time after iteration 10
            print(f"original cell levels: {[x[-1] for x in cell_levels[hour_idx]]}")
            print(f"original prices: {prices[hour_idx]}")

            total_profit += profit
            total_regret += regret
            prices[hour_idx] = new_prices
            cell_levels[hour_idx] = new_cell_levels
            temperature = temperature*(annealing_alpha**annealing_steps)

            print(f"new cell levels: {[x[-1] for x in cell_levels[hour_idx]]}")
            print(f"new prices: {prices[hour_idx]}")

            for cell_idx in range(n_cells):
                s_in_cell = len(cell_to_station[cell_idx])
                ending_cell_levels[hour_idx][cell_idx] = last_vector_iter[cell_idx][-s_in_cell:]
        
        iter_profits.append(total_profit)
        iter_reb_costs.append(reb_cost)
        iter_regret.append(total_regret)

        print(f"At iteration {iteration}...profit: {total_profit}, regret: {total_regret}, rebalancing costs: {reb_cost}")

        print(f"previous iterations: {iter_profits}")
        print(f".with rebalancing costs: {iter_reb_costs}")
        print(f"and regret: {iter_regret}")
        
        iteration += 1

    

if __name__ == "__main__":
    stations = pd.read_csv(f"{data_folder}/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = int(stations["cell"].max())+1
    n_stations = len(list(stations["index"]))
    cell_to_station = [[] for i in range(n_cells)]
    #station_to_cell = [0 for i in range(n_stations)]
    station_to_cell = {}
    capacities = [[] for i in range(n_cells)]
    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])
        capacities[int(row["cell"])].append(int(row["capacity"]))

    # start_hour, end_hour, first_vec_iter, subsidies

    tic = time.perf_counter()
    #res, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, profit, regret = run_control_period(5,20,"none", prices)
    #price_search()

    rebalancing_cost = float(sys.argv[1])
    bounce_cost = float(sys.argv[2])

    optimize_start(rebalancing_cost, bounce_cost, run_price=False)##, default_epoch=default_epoch, default_prices=default_prices)

    toc = time.perf_counter()
    print(f"time diff: {toc-tic}")
