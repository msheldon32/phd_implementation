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
import time
import copy

import scipy.integrate as spi


# Parameters
TIME_POINTS_PER_HOUR = 100
ATOL = 10**(-6)


class SuperCell:
    def __init__(self, model, children="none"):
        self.children = []
        if self.children != "none":
            self.children = children
        self.n_cells = len(self.children)
        self.model = model
    
    def solve_iteration(self, n_entries, configuration, starting_vector, ode_method, epsilon, selected_cells="all"):
        print("Running Trajectory-Based Iteration")

        if selected_cells == "all":
            selected_cells = [i for i in range(self.n_cells)]

        all_res = []

        tic = time.perf_counter()

        cur_epsilon = 0

        x_res = []

        n_time_points = configuration.time_end*TIME_POINTS_PER_HOUR
        time_points = [(i*configuration.time_end)/n_time_points for i in range(n_time_points+1)]
        trajectories = [[np.zeros(n_time_points+1) for j in range(self.model.n_stations)] for i in range(self.model.n_stations)]
        
        for traj_cell in self.children:
            traj_cell.set_timestep(configuration.time_end/n_time_points)

        station_vals = []

        n_iterations = 0

        non_h_idx = []

        for iter_no in range(configuration.max_iterations):
            n_iterations = iter_no + 1

            for tc in self.children:
                tc.set_trajectories(trajectories)

            new_trajectories = copy.deepcopy(trajectories)

            x_iter = np.array([[0.0 for i in range(n_time_points+1)] for i in range(n_entries)])

            for cell_idx in selected_cells:
                traj_cell = self.children[cell_idx]
                x_t = spi.solve_ivp(traj_cell.dxdt_array, [0, configuration.time_end], starting_vector[cell_idx], 
                                        t_eval=time_points, 
                                        method=ode_method, atol=ATOL)
                

                for i, src_stn in enumerate(traj_cell.stations):
                    sy_idx = traj_cell.get_station_idx(i)
                    
                    for dst_stn in range(self.model.n_stations):
                        y_idx = traj_cell.get_delay_idx(i, dst_stn)
                        new_trajectories[src_stn][dst_stn] = x_t.y[y_idx, :]
                
                x_iter[traj_cell.get_idx(), :] = x_t.y
            
            x_res.append(x_iter)

            trajectories = new_trajectories

            if iter_no > 0:
                error_score = (abs(x_res[-1] - x_res[-2])).max()
            else:
                error_score = abs(x_res[-1]).max()

            if error_score < epsilon:
                break
            
            print(f"Iteration complete, time: {time.perf_counter()-tic}, error: {error_score}")
        toc = time.perf_counter()
        print(f"Trajectory-Based Iteration finished, time: {toc-tic}.")

        return [time_points, x_res[-1], toc-tic]

    def solve_discrete_step(self, configuration, starting_vector, ode_method, step_size, selected_cells="all"):
        tic = time.perf_counter()

        if selected_cells == "all":
            selected_cells = [i for i in range(self.n_cells)]


        last_states = [[0 for j in range(self.model.n_stations)] for i in range(self.model.n_stations)]

        time_points = []
        station_vals = []


        current_vector = copy.deepcopy(starting_vector)

        t = 0


        while t < configuration.time_end:
            sub_time_points = [t+(i*(step_size/configuration.steps_per_dt)) for i in range(configuration.steps_per_dt)]
            if len(sub_time_points) == 0:
                sub_time_points.append(t)
            time_points += sub_time_points
            
            new_lstates = copy.deepcopy(last_states)
            new_vector = copy.deepcopy(current_vector)

            for cell_idx in selected_cells:
                traj_cell = self.children[cell_idx]
                # stop here..
                traj_cell.set_last_states(last_states)

                x_t = spi.solve_ivp(traj_cell.dxdt_const, [t, t+step_size], current_vector[cell_idx], 
                                        t_eval = sub_time_points,
                                        method=ode_method, atol=ATOL)

                for i, src_stn in enumerate(traj_cell.stations):
                    sy_idx = traj_cell.get_station_idx(i)
                    
                    station_vals[src_stn] += [comp.get_traj_fn(x_t.t, x_t.y[sy_idx,:])(t) for t in sub_time_points]
                    
                    new_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

                    for dst_stn in range(self.model.n_stations):
                        y_idx = traj_cell.get_delay_idx(i, dst_stn)
                        last_val = float(x_t.y[y_idx, -1])
                        new_vector[cell_idx][y_idx] = last_val
                        new_lstates[src_stn][dst_stn] = last_val

            last_states = new_lstates
            current_vector = new_vector

            t += step_size

        toc = time.perf_counter()
        print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
        return [time_points, toc-tic]

def get_cell_idx(cell_idx, n_stations, cell_to_station):
    out_list = []

    stations = list(cell_to_station[cell_idx])

    for j, src_station in enumerate(stations):
        for dst_station in range(n_stations):
            out_list.append(src_station*n_stations + dst_station)

    out_list += [(n_stations**2) + i for i in stations]
    return out_list

class TrajCell:
    def __init__(self, cell_idx, station_to_cell, cell_to_station, stations, durations, demands):
        self.cell_idx = cell_idx
        self.n_cells = len(cell_to_station)
        self.n_stations = len(durations) # n stations in the whole model

        self.s_in_cell = len(stations)
        self.stations = stations

        self.station_rev = {
            stn: i for i, stn in enumerate(self.stations)
        }

        self.durations = durations
        self.demands = demands

        self.station_to_cell = station_to_cell
        self.cell_to_station = cell_to_station
    
        self.trajectories = []
        self.last_states = []

        self.total_rate = [0 for i in range(self.n_stations)]

        for j, station_idx in enumerate(self.stations):
            for dest_station in range(self.n_stations):
                self.total_rate[j] += self.demands[station_idx][dest_station]
        
        self.tstep = 0
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories

    def set_last_states(self, last_states):
        self.last_states = last_states
    
    def get_delay_idx(self, i_idx, d_idx):
        # i_idx: index of the source station in self.stations (e.g. the internal index)
        # d_idx: overall index of the destination station
        return (i_idx*self.n_stations) + d_idx

    def get_station_idx(self, i_idx):
        # i_idx: index of the source station in self.stations (e.g. the internal index)
        return (self.n_stations * self.s_in_cell) + i_idx 


    def set_timestep(self, tstep):
        self.tstep = tstep


    def get_idx(self):
        out_list = []

        for j, src_station in enumerate(self.stations):
            for dst_station in range(self.n_stations):
                out_list.append(src_station*self.n_stations + dst_station)

        out_list += [(self.n_stations**2) + i for i in self.stations]
        return out_list

    def dxdt_const(self, t, x):
        deriv = [0 for i in range((self.n_stations*self.s_in_cell)+self.s_in_cell)]
        
        # inline re-implementation for speed
        #get_delay_idx = lambda i_idx, d_idx: (i_idx*self.n_stations) + d_idx
        #get_station_idx = lambda i_idx: (self.n_stations * self.s_in_cell) + i_idx

        # derive output delays
        for j, src_station in enumerate(self.stations):
            for dst_station in range(self.n_stations):
                    #d_idx = get_delay_idx(j, dst_station)
                    #d_idx = (j*self.n_stations) + dst_station
                    #src_idx = (self.n_stations*self.s_in_cell) + j
                    deriv[(j*self.n_stations) + dst_station] += self.demands[src_station][dst_station]*min(x[(self.n_stations*self.s_in_cell) + j], 1)
                    deriv[(j*self.n_stations) + dst_station] -= (1/self.durations[src_station][dst_station])*x[(j*self.n_stations) + dst_station]

        
        # derive station levels
        for j, dst_station in enumerate(self.stations):
            d_idx = (self.n_stations*self.s_in_cell)+j

            for src_station in range(self.n_stations):
                rate = 1/self.durations[src_station][dst_station]
                if src_station in self.cell_to_station[self.cell_idx]:
                    # the source is internal to the cell, so we can solve it with x
                    #corres_idx = self.stations.index(src_station)
                    #corres_idx = self.station_rev[src_station]
                    #src_d_idx = get_delay_idx(corres_idx, dst_station)
                    #src_d_idx = (self.station_rev[src_station]*self.n_stations) + dst_station
                    deriv[d_idx] += rate*x[(self.station_rev[src_station]*self.n_stations) + dst_station]
                else:
                    deriv[d_idx] += rate*self.last_states[src_station][dst_station]
            
            deriv[d_idx] -= self.total_rate[j]*min(x[d_idx],1)

        return deriv

    def dxdt_array(self, t, x):
        deriv = [0 for i in range((self.n_stations*self.s_in_cell)+self.s_in_cell)]
        
        # inline re-implementation for speed
        #get_delay_idx = lambda i_idx, d_idx: (i_idx*self.n_stations) + d_idx
        #get_station_idx = lambda i_idx: (self.n_stations * self.s_in_cell) + i_idx

        # derive output delays
        for j, src_station in enumerate(self.stations):
            for dst_station in range(self.n_stations):
                    #d_idx = get_delay_idx(j, dst_station)
                    #d_idx = (j*self.n_stations) + dst_station
                    #src_idx = (self.n_stations*self.s_in_cell) + j
                    deriv[(j*self.n_stations) + dst_station] += self.demands[src_station][dst_station]*min(x[(self.n_stations*self.s_in_cell) + j], 1)
                    deriv[(j*self.n_stations) + dst_station] -= (1/self.durations[src_station][dst_station])*x[(j*self.n_stations) + dst_station]

        
        # derive station levels
        for j, dst_station in enumerate(self.stations):
            d_idx = (self.n_stations*self.s_in_cell)+j

            for src_station in range(self.n_stations):
                rate = 1/self.durations[src_station][dst_station]
                if src_station in self.cell_to_station[self.cell_idx]:
                    # the source is internal to the cell, so we can solve it with x
                    #corres_idx = self.stations.index(src_station)
                    #corres_idx = self.station_rev[src_station]
                    #src_d_idx = get_delay_idx(corres_idx, dst_station)
                    #src_d_idx = (self.station_rev[src_station]*self.n_stations) + dst_station
                    deriv[d_idx] += rate*x[(self.station_rev[src_station]*self.n_stations) + dst_station]
                else:
                    #t_point = int(t//self.tstep)
                    #print(self.trajectories[src_station][dst_station])
                    #print(self.trajectories[src_station][dst_station][0])
                    #print(t_point)
                    deriv[d_idx] += rate*self.trajectories[src_station][dst_station][int(t//self.tstep)]
                    #deriv[d_idx] += rate*self.trajectories[src_station][dst_station][]
            
            deriv[d_idx] -= self.total_rate[j]*min(x[d_idx],1)

        return deriv

    def dxdt_func(self, t, x):
        deriv = [0 for i in range((self.n_stations*self.s_in_cell)+self.s_in_cell)]
        
        # inline re-implementation for speed
        #get_delay_idx = lambda i_idx, d_idx: (i_idx*self.n_stations) + d_idx
        #get_station_idx = lambda i_idx: (self.n_stations * self.s_in_cell) + i_idx

        # derive output delays
        for j, src_station in enumerate(self.stations):
            for dst_station in range(self.n_stations):
                    #d_idx = get_delay_idx(j, dst_station)
                    #d_idx = (j*self.n_stations) + dst_station
                    #src_idx = (self.n_stations*self.s_in_cell) + j
                    deriv[(j*self.n_stations) + dst_station] += self.demands[src_station][dst_station]*min(x[(self.n_stations*self.s_in_cell) + j], 1)
                    deriv[(j*self.n_stations) + dst_station] -= (1/self.durations[src_station][dst_station])*x[(j*self.n_stations) + dst_station]

        
        # derive station levels
        for j, dst_station in enumerate(self.stations):
            d_idx = (self.n_stations*self.s_in_cell)+j

            for src_station in range(self.n_stations):
                rate = 1/self.durations[src_station][dst_station]
                if src_station in self.cell_to_station[self.cell_idx]:
                    # the source is internal to the cell, so we can solve it with x
                    #corres_idx = self.stations.index(src_station)
                    #corres_idx = self.station_rev[src_station]
                    #src_d_idx = get_delay_idx(corres_idx, dst_station)
                    #src_d_idx = (self.station_rev[src_station]*self.n_stations) + dst_station
                    deriv[d_idx] += rate*x[(self.station_rev[src_station]*self.n_stations) + dst_station]
                else:
                    deriv[d_idx] += rate*self.trajectories[src_station][dst_station](t)
                    #deriv[d_idx] += rate*self.trajectories[src_station][dst_station][]
            
            deriv[d_idx] -= self.total_rate[j]*min(x[d_idx],1)

        return deriv


def get_traj_fn_decay(lval, t0, decay):
    def traj_fn(t):
        return lval*math.exp(-decay*(t-t0))
    return traj_fn

def get_traj_fn_lval(lval):
    def traj_fn(t):
        return lval
    return traj_fn

def get_traj_fn(traj_t, traj_x):
    traj_map = {
        t: x
            for t, x in zip(traj_t, traj_x)
    }

    def traj_fn(t):
        if t == 0:
            return traj_x[0]
        elif t in traj_map:
            return traj_map[t]
        np_idx = np.searchsorted(traj_t, t)
        after_t = traj_t[np_idx]
        if after_t == t or np_idx == 0:
            return traj_x[np_idx]
        prev_t = traj_t[np_idx-1]
        after_weight = (t-prev_t)/(after_t-prev_t)
        return traj_x[np_idx-1]*(1-after_weight) + traj_x[np_idx]*(after_weight)
    return traj_fn
