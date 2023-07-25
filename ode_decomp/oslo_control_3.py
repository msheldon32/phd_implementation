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

import gurobipy as gp
from gurobipy import GRB


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

data_folder = "oslo_data_4"
out_folder = "oslo_out"

# Parameters
TIME_POINTS_PER_HOUR = 100
ATOL = 10**(-5)
RATE_MULTIPLIER = 1
SOLVER = "RK45"
TEST_PARAM = False
REQUIRE_POS_GRAD = False
CAPACITY = 15
N_PERIODS = 12 
PRICE_X_THRESHOLD = 0.5


MAX_PRICE = 1.5
MIN_PRICE = 0.5

MAX_PRICE_STEP = 0.3

#random.seed(300)

# how to control the price weighing within a cell
# "proportionate": increases go first to the least loaded station
# "equal": same price across all stations within cell
# "inbound": price determines demand into a cell rather than out.
PRICE_IN_CELL = "equal" 


def run_control(model_data, traj_cells, ode_method, epsilon, cell_limit=False, cell_inc="none", current_vector="none", trajectories="none", prior_res="none", time_length="default", max_steps=100):
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
                            print(f"adding in cell number {next_cell}")
                            print(f"difference: {abs(np.frombuffer(twrap[next_cell][cell_idx*2 + phase_idx].get_obj())[:] - trajectories[next_cell][cell_idx*2 + phase_idx]).max()}")
                            included_cells[next_cell] = 1
                            added = True
                            break


        finished_cells[cell_idx] = True
    


    for iter_no in range(min(model_data.max_iterations, max_steps)):
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
        
        toc = time.perf_counter()
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

def run_control_period_sa(start_hour, end_hour, start_level, prices, final_cell_levels):
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

    delay_phase_ct = [sum([len(x) for x in model_data.mu[0][i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i

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

    for i in range(5): 
        for i, traj_cell in enumerate(traj_cells):
            if PRICE_IN_CELL == "proportionate":
                traj_cell.set_price_proportionate(prices[i], start_level[i], MIN_PRICE, MAX_PRICE)
            elif PRICE_IN_CELL == "inbound":
                traj_cell.set_inbound_prices(prices)
                traj_cell.use_inbound_price = True
            else:
                traj_cell.set_prices(prices[i])

        # run initial model
        # all_res, x_res[-1], trajectories, out_vector, total_reward, profits, regret, arrivals
        ares, lastres, trajectories, last_vector_iter, total_reward, profits, regret, arrivals, bounces = run_control(model_data, traj_cells, SOLVER, 0.5, current_vector=start_level, time_length=float(end_hour-start_hour), max_steps=10)
        gc.collect()

        cell_starting_levels = np.array([sum(start_level[x] for x in station_to_cell[i]) for i in range(n_stations)])
        h_levels = np.array([traj_cells[i].H for i in range(n_cells)])
        l_levels = np.array([traj_cells[i].L for i in range(n_cells)])
        total_cell_flows, partial_cell_flows = spatial_decomp_strict.get_cell_flows(trajectories, N_PERIODS, n_cells)

        high_level = gp.Model("high_level")
        price_multipliers = high_level.addVars(n_cells, lb=MIN_PRICE, ub=MAX_PRICE, name="price_multipliers")
        start_multipliers = high_level.addVars(n_cells, lb=0, ub=5, name="start_multipliers")

        m.setObjective(((2*(price_multipliers @ total_cell_flows)) -
                        (((price_multipliers ** 2) @ total_cell_flows))).sum(), GRB.MAXIMIZE)

        m.addConstr((2-prices) @ total_cell_flows == ((2-prices) * np.identity(n_cells)) @ total_cell_flows @ np.ones(n_stations))

        for period in range(N_PERIODS):
            m.addConstr((cell_starting_levels*start_multipliers) + (2-price_multipliers) @ partial_cell_flows[period] -
                            ((2-price_multipliers) * np.identity(n_cells)) @ partial_cell_flows[period] @ np.ones(n_cells) <= h_levels)
            m.addConstr((cell_starting_levels*start_multipliers) + (2-price_multipliers) @ partial_cell_flows[period] -
                            ((2-price_multipliers) * np.identity(n_cells)) @ partial_cell_flows[period] @ np.ones(n_cells) >= l_levels)

        m.optimize()
        
        start_multipliers = np.array([start_multipliers[i].x for i in range(n_cells)])
        price_multipliers = np.array([price_multipliers[i].x for i in range(n_cells)])

        prices = [[price_multipliers[cell_idx] * y for y in prices[cell_idx]] for cell_idx in range(n_cells)]
        start_level = [[start_multipliers[cell_idx] * y for y in start_level[cell_idx]] for cell_idx in range(n_cells)]


    # optimize flow between cells

    profits = [x - (bounce_cost*y) for x,y in zip(profits, bounces)]

    reb_cost = 0

    for cell_idx in range(n_cells):
        for stn_idx in range(len(list(cell_to_station[cell_idx]))):
            final_level = last_vector_iter[cell_idx][-1-stn_idx]
            if final_cell_levels == "same":
                if change_one:
                    raise Exception("not implemented")
                else:
                    next_level = start_level[cell_idx][stn_idx]
            else:
                next_level = final_cell_levels[cell_idx][stn_idx]
            reb_cost += rebalancing_cost*abs(final_level - next_level)

    for cell_idx in range(n_cells):
        p, x = traj_cells[cell_idx].find_optimum_start_price()
        prices[cell_idx] = p
        starting_levels[cell_idx] = x

    print(f"reward: {total_reward}, bounces: {sum(bounces)}, profits: {sum(profits)}, rebalancing: {reb_cost}")
    total_reward -= bounce_cost*(sum(bounces))

        
    return [last_vec_iter, start_level, prices, total_reward, profits, regret, reb_cost]



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
     
        
def optimize_start(rebalancing_cost, bounce_cost, run_price=True, run_xdiff=True):
    # misnomer, this is actually the overall simulated annealing optimization function
    start_hours = [5]
    hour_delta = 16

    annealing_steps = 5 # formerly 10
    temperature = 20
    annealing_alpha = 0.98

    #raise Exception("price/start change")
    #temperature = temperature * (annealing_alpha**(130))

    


    iteration = 0

    starting_level = 10.0
    prices = [[[1.0 for i in range(len(cell_to_station[cell_idx]))] for cell_idx in range(n_cells)] for hr in start_hours]
    start_levels = [[[10 for i in range(len(cell_to_station[cell_idx]))] for cell_idx in range(n_cells)] for hr in start_hours]

    delay_phase_ct = get_delay_phase_ct(start_hours[0], start_hours[-1] + hour_delta)

    ending_cell_levels = copy.deepcopy(cell_levels)

    iter_profits   = []
    iter_reb_costs = []
    iter_regret    = []


    while True:
        total_profit = 0
        total_regret = 0

        for hour_idx, start_hour in enumerate(start_hours):
            print(f"analyzing {hour_idx}")
            end_hour = start_hour + hour_delta
            n_hours = len(start_hours)

            print(f"original prices: {prices[hour_idx]}")
            print(f"original cell_levels: {[x for x in cell_levels[hour_idx]]}")

            # in: start_hour, end_hour, first_vec_iter, prices, cell_levels, prior_cell_levels, final_cell_levels,
            last_vector_iter, start_levels[hour_idx], prices[hour_idx], profit, profits, regret, reb_cost = run_control_period_sa(start_hour,end_hour,copy.deepcopy(start_levels[hour_idx]),copy.deepcopy(prices[hour_idx]), "same")

            total_profit += profit
            total_regret += regret

            #print(f"new station_levels: {station_levels}")
            print(f"new prices: {prices[hour_idx]}")
            print(f"new cell_levels: {[x for x in cell_levels[hour_idx]]}")

            for cell_idx in range(n_cells):
                s_in_cell = len(cell_to_station[cell_idx])
                start_levels[hour_idx][cell_idx] = last_vector_iter[cell_idx][-s_in_cell:]
        
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
