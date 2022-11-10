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

data_folder = "oslo_data_3"
out_folder = "oslo_out"

# Parameters
TIME_POINTS_PER_HOUR = 100
ATOL = 10**(-6)
RATE_MULTIPLIER = 1
SOLVER = "RK45"


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
        trajectories = [[np.zeros(n_time_points+1) for start_cell in range(traj_cells[end_cell].in_offset)] for end_cell in range(model_data.n_cells)]
    
    if prior_res != "none":
        x_res = [prior_res]
    
    for traj_cell in traj_cells:
        traj_cell.set_timestep(time_length/n_time_points)

    station_vals = []

    n_iterations = 0

    non_h_idx = []

    last_iter = False

    def limited_cell_fn(cell_idx, trajwrap, iterwrap, profitwrap):
        istart = x_idx_cell[cell_idx]
        iend = istart + traj_cells[cell_idx].x_size()
        if len(x_res) != 0:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = x_res[-1][istart:iend, :]
        else:
            np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = 0
                
        for next_cell in range(model_data.n_cells):
            for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase]
        finished_cells[cell_idx] = True

    def cell_fn(cell_idx, trajwrap, iterwrap, profitwrap):
        traj_cells[cell_idx].set_trajectories(trajectories[cell_idx])
        
        istart = x_idx_cell[cell_idx]
        iend = istart + traj_cells[cell_idx].x_size()

        x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_array, [0, time_length], list(current_vector[cell_idx]) + [0,0],
                                t_eval=time_points, 
                                method=ode_method, atol=ATOL)
                
        for next_cell in range(model_data.n_cells):
            for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                    
                np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] = x_t.y[phase_qty_idx, :]

        xiterlock.acquire()
        np.frombuffer(iwrap.get_obj()).reshape(x_iter_shape)[istart:iend, :] = x_t.y[:-2,:]
        xiterlock.release()

        profitlock.acquire()
        profitwrap.get_obj()[0] += x_t.y[-1,-1] # profit
        profitwrap.get_obj()[1] += x_t.y[-2,-1] # regret
        profits.get_obj()[cell_idx] = x_t.y[-1,-1]
        profitlock.release()

        if cell_limit:
            if len(x_res) == 0:
                error_score = float("inf")
            else:
                error_score = (abs(x_t.y[:-2,:] - x_res[-1][traj_cells[cell_idx].get_idx(),:])).max()

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
                error_score = (abs(x_t.y[:-2,:] - x_res[-1][istart:iend,:])).max()

            if error_score < epsilon:
                lc_lock.acquire()
                del included_cells[cell_idx]
                lc_lock.release()
            else:
                for next_cell in range(model_data.n_cells):
                    if next_cell == cell_idx or next_cell in included_cells:
                        continue


                    for phase, phase_rate in enumerate(model_data.mu[0][cell_idx][next_cell]):
                        phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                            
                        if abs(np.frombuffer(twrap[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase].get_obj())[:] - trajectories[next_cell][traj_cells[next_cell].x_in_idx[cell_idx] + phase]).max() >= epsilon:
                            print("adding in cell...")
                            included_cells[next_cell] = 1
                            added = True
                            break


        finished_cells[cell_idx] = True
    


    for iter_no in range(model_data.max_iterations):
        n_iterations = iter_no + 1

        gc.collect()

        iwrap = multiprocessing.Array(ctypes.c_double, int(n_entries*(n_time_points+1)))
        twrap = [[multiprocessing.Array(ctypes.c_double, n_time_points+1) for j in range(traj_cells[i].in_offset)] for i in range(model_data.n_cells)]
        pwrap = multiprocessing.Array(ctypes.c_double, 2) 


        cell_threads = [None for i in range(model_data.n_cells)]
        x_t_cell = [None for i in range(model_data.n_cells)]

        x_iter_shape = (n_entries, n_time_points + 1)

        finished_cells = Manager().dict()

        for cell_idx in range(model_data.n_cells):
            lc_lock = Lock()
            xiterlock = Lock()
            profitlock = Lock()
            
            if last_iter:
                # reverse things on the last iteration, as the limited cells need to be ran again
                if cell_idx in limited_cells and limited_cells[cell_idx] < iter_no - 1:
                    cell_threads[cell_idx] = Process(target=cell_fn, 
                        args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock, limited_cells))
                else:
                    cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                        args=(cell_idx, twrap, iwrap, prwap))#, lc_lock, xiterlock))

            elif cell_limit and cell_idx in limited_cells:
                cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                    args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock))
            elif cell_include and cell_idx not in included_cells:
                cell_threads[cell_idx] = Process(target=limited_cell_fn, 
                    args=(cell_idx, twrap, iwrap, pwrap))
            else:
                cell_threads[cell_idx] = Process(target=cell_fn, 
                    args=(cell_idx, twrap, iwrap, pwrap))#, lc_lock, xiterlock, limited_cells))
        
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

        if last_iter:
            toc = time.perf_counter()
            all_res.append([time_points, x_res[-1], toc - tic, n_iterations])
            break

        for end_cell in range(model_data.n_cells):
            for traj_start in range(len(trajectories[end_cell])):
                trajectories[end_cell][traj_start] = np.frombuffer(twrap[end_cell][traj_start].get_obj())

        if len(x_res) == 1:
            error_score = float("inf")
        else:
            error_score = (abs(x_res[-1] - x_res[-2])).max()

        print(f"Reward: {pwrap.get_obj()[0]}, regret: {pwrap.get_obj()[1]}")

        total_reward = pwrap.get_obj()[0]

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
    
    print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

    profits = list(np.frombuffer(profits.get_obj()))

    return [all_res, x_res[-1], trajectories, out_vector, total_reward, profits]


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

def run_control_period(start_hour, end_hour, first_vec_iter, prices,
            finite_difference_x = 0.1, finite_difference_price=0.01, run_price=True, cache=True):
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

    if first_vec_iter == "none":
        delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
        first_vec_iter = [
            [0.0 for i in range(delay_phase_ct[cell_idx])] +
            [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
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
            rev_hr_cell(out_demands,cell_idx))

            for cell_idx in range(n_cells)
    ]
    
    for i, traj_cell in enumerate(traj_cells):
        traj_cell.set_price(prices[i])

    # run initial model
    filename = "control_res/res_{}_{}".format(start_hour,end_hour)

    if os.path.isfile(filename) and cache:
        with open(filename, "rb") as f:
            ares, lastres, trajectories, last_vector_iter, trajectories, total_reward, profits = pickle.load(f)
    else:
        ares, lastres, trajectories, last_vector_iter, total_reward, profits = run_control(model_data, traj_cells, SOLVER, 0.5, current_vector=first_vec_iter, time_length=float(end_hour-start_hour))

        print(f"reward: {total_reward}")

        if cache:
            with open(filename, "xb") as f:
                pickle.dump([ares, lastres, trajectories, last_vector_iter, trajectories, total_reward, profits],f)


    for cell_idx, traj_cell in enumerate(traj_cells):
        for station_cell_idx, station_idx in enumerate(cell_to_station[cell_idx]):
            station_iterres[station_idx] = last_vector_iter[cell_idx][traj_cell.station_offset + station_cell_idx]


    # estimate derivatives based on x_0
    dprofit_dx = []
    dxf_dx = []

    for cell_idx in range(n_cells):
        x0_total = sum(first_vec_iter[cell_idx])
        inf_factor = 1 + (finite_difference_x/x0_total) if x0_total != 0 else 1
        starting_vec = [(x if i != cell_idx else [pt*inf_factor for pt in x]) for i, x in enumerate(first_vec_iter)]
        ares, lastres, sample_trajectories, sample_last_vector, new_total_reward, new_profits = run_control(model_data, traj_cells, SOLVER, 0.05, trajectories=trajectories, 
                        prior_res=lastres, cell_inc=[cell_idx], current_vector=starting_vec, time_length=float(end_hour-start_hour))

        dprofit_dx.append(sum([(x - profits[i])/finite_difference_x for i, x in enumerate(new_profits) if new_profits[i] != 0]))
        dxf_dx.append([((sum(sample_last_vector[i]) - sum(last_vector_iter[i]))/finite_difference_x  if new_profits[i] != 0 else 0) for i in range(n_cells)])
    
    dprofit_dp = []
    dxf_dp = []

    if run_price:
        for cell_idx in range(n_cells):
            traj_cells[cell_idx].set_price(traj_cells[cell_idx].price + finite_difference_price)

            starting_vec = [(x if i != cell_idx else x) for i, x in enumerate(first_vec_iter)]
            ares, lastres, sample_trajectories, sample_last_vector, new_total_reward, new_profits = run_control(model_data, traj_cells, SOLVER, 0.05, trajectories=trajectories, 
                            prior_res=lastres, cell_inc=[cell_idx], current_vector=first_vec_iter, time_length=float(end_hour-start_hour))

            dprofit_dp.append(sum([(x - profits[i])/finite_difference_price for i, x in enumerate(new_profits) if new_profits[i] != 0]))
            dxf_dp.append([((sum(sample_last_vector[i]) - sum(last_vector_iter[i]))/finite_difference_price  if new_profits[i] != 0 else 0) for i in range(n_cells)])

            traj_cells[cell_idx].set_price(traj_cells[cell_idx].price - finite_difference_price)

    for cell_idx, traj_cell in enumerate(traj_cells):
        for station_cell_idx, station_idx in enumerate(cell_to_station[cell_idx]):
            station_iterres[station_idx] = last_vector_iter[cell_idx][traj_cell.station_offset + station_cell_idx]

    with open(f"{out_folder}/res_iter_control_{start_hour}_{end_hour}", "w") as f:
        json.dump(station_iterres, f)
        
    return [station_iterres, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, total_reward, profits]

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

    # to fix: dxf_dx is a vector of partial derivatives not a scalar

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


def get_first_vec_iter(start_hour, end_hour):
    # demands
    in_demands = []
    in_probabilities = []
    out_demands = []
    station_iterres = [0 for i in range(n_stations)]
    station_sampres = [0 for i in range(n_stations)]

    mu = []
    phi = []

    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
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
    delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    first_vec_iter = [
        [0.0 for i in range(delay_phase_ct[cell_idx])] +
        [float(x) for i, x in enumerate(model_data.starting_bps) if i in list(cell_to_station[cell_idx])]
            for cell_idx in range(model_data.n_cells)
    ]
    return first_vec_iter

def optimize_start(rebalancing_cost):
    original_level = 6.0
    vec_iter = get_first_vec_iter(5,20)

    alpha = 0.1
    min_alpha = 0.001
    alpha_decay = 0.99

    iteration = 0

    while True:
        filename = "start_res/res_{}".format(iteration)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                vec_iter, last_vector_iter, new_vec_iter, profit, regret, reb_cost = pickle.load(f)
                
            vec_iter = new_vec_iter
            alpha = ((alpha-min_alpha)*alpha_decay) + min_alpha

            iteration += 1
            continue


        res, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, profit, regret = run_control_period(5,8,vec_iter, prices, run_price=False, cache=False, finite_difference_x=0.3, finite_difference_price=0.5)

        new_vec_iter, reb_cost = start_step(alpha, vec_iter, last_vector_iter, dprofit_dx, dxf_dx, rebalancing_cost)

        print(f"At iteration {iteration}...profit: {profit}, rebalancing costs: {reb_cost}")

        if os.path.isfile(filename):
            with open(filename, "wb") as f:
                pickle.dump([vec_iter, last_vector_iter, new_vec_iter, profit, regret, reb_cost], f)
        else:
            with open(filename, "xb") as f:
                pickle.dump([vec_iter, last_vector_iter, new_vec_iter, profit, regret, reb_cost], f)
        
        vec_iter = new_vec_iter
        alpha = ((alpha-min_alpha)*alpha_decay) + min_alpha

        iteration += 1

def price_step_final(alpha, vec_iter, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, rebalancing_cost, prices):
    """Adapted from start_step for prices. This assumes that we're in the last stage. 
    """

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
            cost_delta += sum([rebalancing_cost*ext*(dxf_dp[cell_idx][other_cell]-indicator) for ext in exp_terms[other_cell]])
        rebalancing_deriv.append(cost_delta)

    # 2. find overall_gradient w.r.t. starting point
    overall_deriv = [pd-rd for rd,pd in zip(rebalancing_deriv, dprofit_dp)]

    # 3. find new prices
    new_prices = []
    for cell_idx in range(n_cells):
        new_prices.append(prices[cell_idx] + alpha*overall_deriv[cell_idx])
        
    # 5. calculate rebalancing cost for the previous iteration
    reb_costs = []
    for cell_idx in range(n_cells):
        xstart = vec_iter[cell_idx]
        xend   = last_vector_iter[cell_idx]
        reb_costs.append(sum([rebalancing_cost*max(xf-x0,0) for x0, xf in zip(xstart, xend)]))
    
    return new_prices, sum(reb_costs)
     
def optimize_price(rebalancing_cost):
    original_level = 6.0
    vec_iter = get_first_vec_iter(5,20)

    alpha = 0.1
    min_alpha = 0.001
    alpha_decay = 0.99

    iteration = 0

    prices = [1.0 for cell_idx in range(n_cells)]

    while True:
        filename = "price_res/res_{}".format(iteration)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                vec_iter, last_vector_iter, profit, new_prices, prices, reb_cost = pickle.load(f)
                
            prices = new_prices
            alpha = ((alpha-min_alpha)*alpha_decay) + min_alpha

            iteration += 1
            continue



        res, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, profit, regret = run_control_period(5,8,vec_iter, prices, cache=False, finite_difference_x=0.3, finite_difference_price=0.5)

        new_prices, reb_cost = price_step_final(alpha, vec_iter, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, rebalancing_cost, prices)

        print(f"At iteration {iteration}...profit: {profit}, rebalancing costs: {reb_cost}")

        if os.path.isfile(filename):
            with open(filename, "wb") as f:
                pickle.dump([vec_iter, last_vector_iter, profit, new_prices, prices, reb_cost], f)
        else:
            with open(filename, "xb") as f:
                pickle.dump([vec_iter, last_vector_iter, profit, new_prices, prices, reb_cost], f)
        
        prices = new_prices
        alpha = ((alpha-min_alpha)*alpha_decay) + min_alpha

        iteration += 1
   
    

if __name__ == "__main__":
    stations = pd.read_csv(f"{data_folder}/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = stations["cell"].max() + 1
    n_stations = stations["index"].max() + 1
    cell_to_station = [[] for i in range(n_cells)]
    station_to_cell = [0 for i in range(n_stations)]
    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])

    # start_hour, end_hour, first_vec_iter, subsidies
    prices = [1.0 for i in range(n_cells)]

    tic = time.perf_counter()
    #res, last_vector_iter, dprofit_dx, dxf_dx, dprofit_dp, dxf_dp, profit, regret = run_control_period(5,20,"none", prices)
    optimize_price(1.0)

    toc = time.perf_counter()
    print(f"time diff: {toc-tic}")