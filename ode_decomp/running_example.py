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

import spatial_decomp_station
import comp
from oslo_data import *
import large_fluid

# Parameters
N_ITERATIONS = 6
N_STATIONS = 4
STATIONS_PER_CELL = 2
ODE_METHOD = "BDF"
N_CELLS = 2
TIME_END = 4
STARTING_BIKES = 20
N_TIME_POINTS = 1000
ATOL = 10**(-6)

class CompExample:
    def __init__(self, durations, demands):
        self.comp_model = comp.CompModel(durations, demands)

        bikes_per_station = STARTING_BIKES/N_STATIONS

        self.starting_vector = [0 for i in range(N_STATIONS**2)] + [bikes_per_station for i in range(N_STATIONS)]

        self.display = False

    def run(self):
        time_points = [(i*TIME_END)/N_TIME_POINTS for i in range(N_TIME_POINTS+1)]
        
        x_t = spi.solve_ivp(self.comp_model.dxdt, [0, TIME_END], self.starting_vector, 
                                t_eval=time_points, method=ODE_METHOD, atol=ATOL)

        if self.display:
            for stn_idx in range(N_STATIONS):
                station_vals = [float(y) for y in x_t.y[self.comp_model.get_station_idx(stn_idx), :]]
                plt.plot(time_points, station_vals, alpha=0.3)
                
                plt.legend([f"iter {iter_no}" for iter_no in range(N_ITERATIONS)])
                plt.xlabel("time")
                plt.ylabel("number of bikes at station")
                plt.title(f"Queue Length at Station {stn_idx}")
                plt.show()
        return x_t

class DiscreteStepEx:
    def __init__(self, durations, demands, step_sizes):
        self.demands = demands
        self.durations = durations
        self.bikes_per_station = STARTING_BIKES/N_STATIONS

        self.starting_vector = [[0 for i in range(STATIONS_PER_CELL*N_STATIONS)] + [self.bikes_per_station,self.bikes_per_station],
                        [0 for i in range(STATIONS_PER_CELL*N_STATIONS)] + [self.bikes_per_station,self.bikes_per_station]]
        station_to_cell = [1,1,2,2]
        cell_to_station = [set([0,1]), set([2,3])]
                
        self.traj_cells = [spatial_decomp_station.TrajCell(0, station_to_cell, cell_to_station, [0,1], durations, demands),
                           spatial_decomp_station.TrajCell(1, station_to_cell, cell_to_station, [2,3], durations, demands)]
        
        self.step_sizes = step_sizes

    def run(self, true_traj):
        station_vals = []
        step_time_points = []
        for step_idx in range(len(self.step_sizes)):
            step_vals = []
            for i in range(N_STATIONS):
                step_vals.append([])
            station_vals.append(step_vals)
            step_time_points.append([])
            
        base_time_points = [(i*TIME_END)/N_TIME_POINTS for i in range(N_TIME_POINTS+1)]

        for step_idx, step_size in enumerate(self.step_sizes):
            #print(f"Running step_size {step_size}")

            n_substeps = math.floor((step_size/TIME_END)*N_TIME_POINTS)

            trajectories = [[(lambda t: 0) for j in range(N_STATIONS)] for i in range(N_STATIONS)]


            x_station = [[] for i in range(N_STATIONS)]

            total_bikes = 0

            current_vector = copy.deepcopy(self.starting_vector)

            t = 0
            
            while t < TIME_END:
                sub_time_points = [t+((i*step_size)/n_substeps) for i in range(n_substeps)]
                if len(sub_time_points) == 0:
                    sub_time_points.append(t)
                step_time_points[step_idx] += sub_time_points
                
                new_trajectories = copy.deepcopy(trajectories)
                new_vector = copy.deepcopy(current_vector)

                for cell_idx in range(N_CELLS):
                    self.traj_cells[cell_idx].set_trajectories(trajectories)

                    x_t = spi.solve_ivp(self.traj_cells[cell_idx].dxdt, [t, t+step_size], current_vector[cell_idx], 
                                            #t_eval=sub_time_points, 
                                            method=ODE_METHOD, atol=ATOL)

                    for i, src_stn in enumerate(self.traj_cells[cell_idx].stations):
                        sy_idx = self.traj_cells[cell_idx].get_station_idx(i)
                        
                        #station_vals[step_idx][src_stn] += [float(y) for y in x_t.y[sy_idx, :]]
                        station_vals[step_idx][src_stn] += [comp.get_traj_fn(x_t.t, x_t.y[sy_idx,:])(t) for t in sub_time_points]
                        
                        new_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

                        for dst_stn in range(N_STATIONS):
                            y_idx = self.traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                            last_val = float(x_t.y[y_idx, -1])
                            new_vector[cell_idx][y_idx] = last_val
                            delay_rate = 1/self.durations[src_stn][dst_stn]
                            #traj = spatial_decomp_station.get_traj_fn_decay(last_val, t+step_size, delay_rate)
                            traj = spatial_decomp_station.get_traj_fn_lval(last_val)
                            new_trajectories[src_stn][dst_stn] = traj

                    total_bikes += sum(x_t.y[:,-1])

                trajectories = new_trajectories
                current_vector = new_vector

                t += step_size
        

        for stn_idx in range(N_STATIONS):
            for step_idx, step_size in enumerate(self.step_sizes):
                plt.plot(step_time_points[step_idx], station_vals[step_idx][stn_idx], alpha=0.5)

            plt.plot(base_time_points, [float(x) for x in true_traj.y[(N_STATIONS**2) + stn_idx, :]], linestyle="--", color="black")
            
            plt.legend([f"Δt = {x}" for x in self.step_sizes] + ["true value"])
            plt.xlabel("time")
            plt.ylabel("number of bikes at station")
            plt.title(f"Queue Length at Station {stn_idx + 1} (Discrete-Step Submodeling)")
            plt.ylim(0,5)
            plt.show()

class TrajectoryIterationEx:
    def __init__(self, durations, demands):
        self.demands = demands
        self.durations = durations
        self.bikes_per_station = STARTING_BIKES/N_STATIONS

        self.starting_vector = [[0 for i in range(STATIONS_PER_CELL*N_STATIONS)] + [self.bikes_per_station,self.bikes_per_station],
                        [0 for i in range(STATIONS_PER_CELL*N_STATIONS)] + [self.bikes_per_station,self.bikes_per_station]]
        station_to_cell = [1,1,2,2]
        cell_to_station = [set([0,1]), set([2,3])]
                
        self.traj_cells = [spatial_decomp_station.TrajCell(0, station_to_cell, cell_to_station, [0,1], durations, demands),
                      spatial_decomp_station.TrajCell(1, station_to_cell, cell_to_station, [2,3], durations, demands)]

    def run(self, true_traj):
        time_points = [(i*TIME_END)/N_TIME_POINTS for i in range(N_TIME_POINTS+1)]
        trajectories = [[(lambda t: 0) for j in range(N_STATIONS)] for i in range(N_STATIONS)]

        station_vals = []

        for iter_no in range(N_ITERATIONS):
            print(f"Running iteration {iter_no}")

            for tc in self.traj_cells:
                tc.set_trajectories(trajectories)

            new_trajectories = copy.deepcopy(trajectories)

            x_station = [[] for i in range(N_STATIONS)]

            total_bikes = 0

            for cell_idx in range(N_CELLS):
                x_t = spi.solve_ivp(self.traj_cells[cell_idx].dxdt, [0, TIME_END], self.starting_vector[cell_idx], 
                                        t_eval=time_points, method=ODE_METHOD, atol=ATOL)

                for i, src_stn in enumerate(self.traj_cells[cell_idx].stations):
                    sy_idx = self.traj_cells[cell_idx].get_station_idx(i)
                    x_station[src_stn] = [float(y) for y in x_t.y[sy_idx, :]]
                    for dst_stn in range(N_STATIONS):
                        y_idx = self.traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                        new_trajectories[src_stn][dst_stn] = spatial_decomp_station.get_traj_fn(x_t.t, x_t.y[y_idx, :])
                total_bikes += sum(x_t.y[:,-1])
            
            station_vals.append(x_station)
            trajectories = new_trajectories

            print(f"Bike loss: {STARTING_BIKES-total_bikes}")
        
        for stn_idx in range(N_STATIONS):
            for iter_no in range(N_ITERATIONS):
                plt.plot(time_points, station_vals[iter_no][stn_idx], alpha=0.5)

            plt.plot(time_points,[float(x) for x in true_traj.y[(N_STATIONS**2) + stn_idx, :]], linestyle="--", color="black")
            
            plt.legend([f"iter {iter_no + 1}" for iter_no in range(N_ITERATIONS)] + ["true value"])
            plt.xlabel("time")
            plt.ylabel("number of bikes at station")
            plt.title(f"Queue Length at Station {stn_idx+1} (Trajectory-Based Iteration)")
            plt.ylim(0,5)
            plt.show()

if __name__ == "__main__":
    durations = [[0.5 for i in range(N_STATIONS)] for j in range(N_STATIONS)]
    demands   = [[0,5,1,2],
                 [6,0,1,1],
                 [1,1,0,9],
                 [2,1,4,0]]
    
    step_sizes = [1, 0.5, 0.2, 0.1, 0.01, 0.001]

    comp_ex = CompExample(durations, demands)
    ti_example = TrajectoryIterationEx(durations, demands)
    ds_example = DiscreteStepEx(durations, demands, step_sizes)

    correct_x = comp_ex.run()
    ds_example.run(correct_x)
    ti_example.run(correct_x)