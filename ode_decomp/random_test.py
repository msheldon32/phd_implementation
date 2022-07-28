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

HOURLY_CLUSTER = [hr for hr in range(0,24)] # map hour->time cluster

def get_rand_data(n_cells, spc_range, dur_range, in_rate_range, out_rate_range, time_clusters = HOURLY_CLUSTER):
    def get_rate(inside):
        if inside:
            return (random.random()*(in_rate_range[1]-in_rate_range[0])) + in_rate_range[0]
        return (random.random()*(out_rate_range[1]-out_rate_range[0])) + out_rate_range[0]
    def get_dur():
        return (random.random()*(dur_range[1]-dur_range[0])) + dur_range[0]

    cluster_dic = {
        "cell": [],
        "station": []
    }
    n_stations = 0
    for cell_idx in range(n_cells):
        stations_in_cell = random.randrange(spc_range[0], spc_range[1])
        for i in range(stations_in_cell):
            cluster_dic["station"].append(n_stations)
            cluster_dic["cell"].append(cell_idx)
            n_stations += 1
    clusters = pd.DataFrame(cluster_dic)
    
    cell_map = [0 for i in range(n_stations)]

    for i, row in clusters.iterrows():
        cell_map[int(row["station"])] = int(row["cell"])


    n_time_clusters = len(set(time_clusters))
    n_stations = clusters["station"].max()+1
    n_cells = clusters["cell"].max()+1


    # [time][start_cell][end_cell]: duration to transit from start to end
    duration_array = [[
        [get_dur() for end_cell in range(n_cells)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]

    # [time][start_cell][station]: total demand from start_cell to station in time
    in_demands = [[[
                0
            for station in range(n_stations)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]
    
    # [time][start_cell][end_cell]: total demand from start_cell to end_cell
    in_demands_cell = [[[0
            for end_cell in range(n_cells)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]
    
    
    # [time][station][end_cell]: total demand from start_cell to station in time
    out_demands = [[[
                0
            for end_cell in range(n_cells)]
            for station in range(n_stations)]
            for time_cluster in range(n_time_clusters)]


    for time_cluster in range(n_time_clusters):
        for start_station in range(n_stations):
            for end_station in range(n_stations):
                start_cell = cell_map[start_station]
                end_cell   = cell_map[end_station]

                rate = get_rate(start_cell == end_cell)
                in_demands[time_cluster][start_cell][end_station] += rate
                in_demands_cell[time_cluster][start_cell][end_cell] += rate
                out_demands[time_cluster][start_station][end_cell] += rate

    # [time][start_cell][station]: probability to transition to station
    in_probabilities = [[[
                in_demands[time_cluster][start_cell][station] / in_demands_cell[time_cluster][start_cell][cell_map[station]]
                    if in_demands_cell[time_cluster][start_cell][cell_map[station]] != 0 else 0
            for station in range(n_stations)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]

    return [clusters, duration_array, in_demands, in_demands_cell, in_probabilities, out_demands]


def run_model():
    # MODEL PARAMETERS
    hour = 12 
    n_iterations = 7
    time_end = 1
    n_hours  = 24

    n_cells = 25
    spc_range = [3, 20]
    in_rate_range = [0,1]
    out_rate_range = [0,0.5]
    dur_range = [0, 1]

    n_time_points = 1000
    time_points = [(i*time_end)/n_time_points for i in range(n_time_points+1)]
    atol = 10**(-6)

    compare = True

    n_clusters = 50

    ode_method="BDF"
    #ode_method="RK45"

    print("loading data...")
    clusters, duration_array, in_demands, in_demands_cell, in_probabilities, out_demands = get_rand_data(n_cells, spc_range, dur_range, in_rate_range, out_rate_range)
    print("data loaded.")

    print("preprocessing..")
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
    comp_time = 0
    if compare:
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

        comp_time = toc-tic
        print(f"Comparison finished. Time: {toc-tic} seconds")

    print("starting analysis..")

    total_solution_time = 0
    iter_mapes = [[0,0]]
    iter_times = []

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
        iter_times.append(total_solution_time/comp_time)

    return iter_mapes[:-1], iter_times

if __name__ == "__main__":
    all_mapes = []
    all_times = []
    
    fig, ax = plt.subplots(nrows=1, ncols=2)

    n_runs = 10
    for run_no in range(n_runs):
        mape_traj, time_traj = run_model()
        all_mapes.append(mape_traj)
        all_times.append(time_traj)
    iter_x = [i for i in range(2, 8)]
    
    for mape_traj in all_mapes:
        ax[0].plot(iter_x, mape_traj[1:])
    ax[0].set_title("MAPE decrease over time")
    
    for time_traj in all_times:
        ax[1].plot(iter_x, time_traj[1:])
    ax[1].set_title("Iteration time ratio")

    max_time = max([max(x) for x in all_times])

    ax[1].set_yticks([i for i in range(0, math.ceil(max_time))])


    plt.show()