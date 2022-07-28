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

import spatial_decomp
from oslo_data import *
import large_fluid


if __name__ == "__main__":
    # MODEL PARAMETERS
    hour = 12 
    n_iterations = 100
    time_end = 1
    n_hours  = 24

    n_time_points = 1000
    time_points = None#[(i*time_end)/n_time_points for i in range(n_time_points+1)]
    atol = 10**(-6)

    ode_method="BDF"
    #ode_method="RK45"

    print("loading data...")
    clusters, duration_array, in_demands, in_demands_cell, in_probabilities, out_demands = get_oslo_data()
    print("data loaded.")

    print("preprocessing..")
    n_cells = clusters["cell"].max()+1
    n_stations = clusters["station"].max()+1


    #-----------------------------------------------------------------------------------------------------#
    traj_cells         = [[-1 for i in range(n_cells)] for hr in range(n_hours)]
    large_fluid_models = [-1 for hr in range(n_hours)]
    for hr in range(n_hours):
        for cell_idx in range(n_cells):
            stations = list(clusters[clusters["cell"] == cell_idx]["station"])
            traj_cells[hr][cell_idx] = spatial_decomp.TrajCell(cell_idx, stations, duration_array[hr], in_demands_cell[hr],
                    in_probabilities[hr], out_demands[hr])

        large_fluid_models[hr] = large_fluid.LargeFluid(n_cells, clusters, duration_array[hr], in_probabilities[hr], out_demands[hr])

    #---------------------------------------------------------------------------------------------------
    


    # intercell trajectories are labeled [arrival_cell][departure_cell]
    intercell_trajectories = [[(lambda t: 0) for i in range(n_cells)] for j in range(n_cells)]
    in_cell_trajectories = [[] for j in range(n_cells)]

    overall_delay_traj   = [[(lambda t: 0) for i in range(n_cells)] for j in range(n_cells)]
    overall_station_traj = [(lambda t: 0) for i in range(n_stations)]


    bikes_per_cell  = 100
    starting_vector = []
    overall_starting_vector = [0 for i in range((n_cells**2) + n_stations)]

    for cell_idx in range(n_cells):
        stations_in_cell = list(clusters[clusters["cell"] == cell_idx]["station"])
        n_stations_in_cell = len(stations_in_cell)
        cell_start = [0 for i in range(n_cells)] + [(bikes_per_cell/n_stations_in_cell) for i in range(n_stations_in_cell)]
        starting_vector.append(cell_start)
        
        for station_idx in stations_in_cell:
            overall_starting_vector[(n_cells**2)+station_idx] = (bikes_per_cell/n_stations_in_cell)
        
        in_cell_trajectories[cell_idx] = [(lambda t: 0) for i in range(n_cells+n_stations_in_cell)]


    print("preprocessing finished")

    #print("debugging..")
    #for cell_idx in range(n_cells):
    #    for dest_cell in range(n_cells):
    #        in_prob = 0
    #        for station_idx in traj_cells[hour][cell_idx].stations:
    #            in_prob += traj_cells[hour][cell_idx].in_probabilities[source_cell][station_idx]
    #        print(f"In prob: {in_prob}")


    print("comparison model started..")

    tic = time.time()
    x_t = spi.solve_ivp(large_fluid_models[hour].dxdt, [0,time_end], overall_starting_vector, t_eval=time_points, method=ode_method, atol=atol)
    toc = time.time()

    for arrival_cell in range(n_cells):
        for departure_cell in range(n_cells):
            delay_idx = (departure_cell*n_cells)+arrival_cell
            overall_delay_traj[arrival_cell][departure_cell] = spatial_decomp.get_traj_fn(x_t.t, x_t.y[delay_idx, :])
    for station_idx in range(n_stations):
        y_idx = (n_cells**2) + station_idx
        overall_station_traj[station_idx] = spatial_decomp.get_traj_fn(x_t.t, x_t.y[y_idx, :])


    print(f"Comparison finished. Time: {toc-tic} seconds")

    iter_mapes = []
    print("starting analysis..")

    total_solution_time = 0
    iter_mapes = [[0,0]]

    for iter_no in range(n_iterations):
        print(f"Running iteration {iter_no}")
        x_cell = []

        bike_loss = sum([sum(starting_vector[cell_idx]) for cell_idx in range(n_cells)])

        for cell_idx in range(n_cells):
            print(f"Analyzing cell {cell_idx}")
            stations_in_cell = list(clusters[clusters["cell"] == cell_idx]["station"])
            n_stations_in_cell = len(stations_in_cell)
            traj_in = intercell_trajectories[cell_idx]
            traj_cells[hour][cell_idx].set_trajectories(traj_in)

            tic = time.time()
            #traj_cells[hour][cell_idx].first_iteration = False
            x_t = spi.solve_ivp(traj_cells[hour][cell_idx].dxdt, [0,time_end], starting_vector[cell_idx], t_eval=time_points, method=ode_method, atol=atol) 
            traj_cells[hour][cell_idx].first_iteration = False
            toc = time.time() 

            total_solution_time += toc-tic
            x_cell.append([[float(x) for x in list(x_t.t)], [[float(y) for y in row] for row in list(x_t.y)]])


            total_loss = 0
            total_in_cell_overall = 0
            total_in_cell_decomp  = 0

            mape = [0,0]

            for next_cell in range(n_cells):
                intercell_trajectories[next_cell][cell_idx] = spatial_decomp.get_traj_fn(x_t.t, x_t.y[next_cell, :])
            
            for queue_idx in range(0, n_cells + n_stations_in_cell):
                queue_traj = spatial_decomp.get_traj_fn(x_t.t, x_t.y[queue_idx, :])
                if queue_idx < n_cells:
                    if not compare:
                        baseline = in_cell_trajectories[cell_idx][queue_idx](time_end)
                    else:
                        baseline = overall_delay_traj[queue_idx][cell_idx](time_end)
                else:
                    if not compare:
                        baseline = in_cell_trajectories[cell_idx][queue_idx](time_end)
                    else:
                        baseline = overall_station_traj[stations_in_cell[queue_idx-n_cells]](time_end)
                acc = abs(queue_traj(time_end) - baseline)/baseline
                mape[0] += acc
                mape[1] += 1
                iter_mapes[-1][0] += acc
                iter_mapes[-1][1] += 1
                total_loss += abs(queue_traj(time_end) - baseline)
                total_in_cell_overall += baseline
                total_in_cell_decomp += queue_traj(time_end)
                in_cell_trajectories[cell_idx][queue_idx] = queue_traj 

            print(f"total loss: {total_loss}")
            print(f"cell delta: {total_in_cell_decomp-total_in_cell_overall}")
            print(f"MAPE: {mape[0]/mape[1]}")
            bike_loss -= total_in_cell_decomp
        iter_mapes[-1] = iter_mapes[-1][0] / iter_mapes[-1][1]
        print(f"Iteration finished. Time: {total_solution_time} vs {comp_time}. Lost bikes: {bike_loss}. MAPE: {iter_mapes[-1]}")
        iter_mapes.append([0,0])
        
    print(f"Analysis finished. Time: {total_solution_time}")
    with open("xdump.json", "w") as out_json:
        json.dump(x_cell, out_json)