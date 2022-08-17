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
from oslo_data import *
import large_fluid

if __name__ == "__main__":
    # parameters
    n_iterations = 7
    n_stations = 4
    ode_method = "BDF"

    n_cells = 2
    time_end = 4

    starting_vector = [[0 for i in range(2*4)] + [5,5],
                       [0 for i in range(2*4)] + [5,5]]
    starting_bikes = 20

    n_time_points = 1000
    time_points = [(i*time_end)/n_time_points for i in range(n_time_points+1)]
    atol = 10**(-6)

    station_to_cell = [1,1,2,2]
    cell_to_station = [set([0,1]), set([2,3])]

    durations = [[0.5 for i in range(n_stations)] for j in range(n_stations)]
    demands = [[0,5,1,2],
               [6,0,1,1],
               [1,1,0,9],
               [2,1,4,0]]

    in_prob1 = [[0.5,0.5],[2/3,1/3]]
    out_dem1 = [[1,2],[1,1]]

    in_prob2 = [[]]
    
    traj_cells = [spatial_decomp_station.TrajCell(0, station_to_cell, cell_to_station, [0,1], durations, demands),
                  spatial_decomp_station.TrajCell(1, station_to_cell, cell_to_station, [2,3], durations, demands)]

    
    trajectories = [[(lambda t: 0) for j in range(n_stations)] for i in range(n_stations)]

    station_vals = []

    for iter_no in range(n_iterations):
        print(f"Running iteration {iter_no}")

        for tc in traj_cells:
            tc.set_trajectories(trajectories)

        new_trajectories = copy.deepcopy(trajectories)

        x_station = [[] for i in range(n_stations)]

        total_bikes = 0

        for cell_idx in range(n_cells):
            x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt, [0, time_end], starting_vector[cell_idx], 
                                    t_eval=time_points, method=ode_method, atol=atol)

            for i, src_stn in enumerate(traj_cells[cell_idx].stations):
                sy_idx = traj_cells[cell_idx].get_station_idx(i)
                x_station[src_stn] = [float(y) for y in x_t.y[sy_idx, :]]
                for dst_stn in range(n_stations):
                    y_idx = traj_cells[cell_idx].get_delay_idx(i, dst_stn)
                    new_trajectories[src_stn][dst_stn] = spatial_decomp_station.get_traj_fn(x_t.t, x_t.y[y_idx, :])
            total_bikes += sum(x_t.y[:,-1])
        
        station_vals.append(x_station)
        trajectories = new_trajectories

        print(f"Bike loss: {starting_bikes-total_bikes}")
    
    for stn_idx in range(n_stations):
        for iter_no in range(n_iterations):
            plt.plot(time_points, station_vals[iter_no][stn_idx], alpha=0.3)
        
        plt.legend([f"iter {iter_no}" for iter_no in range(n_iterations)])
        plt.xlabel("time")
        plt.ylabel("number of bikes at station")
        plt.title(f"Queue Length at Station {stn_idx}")
        plt.show()