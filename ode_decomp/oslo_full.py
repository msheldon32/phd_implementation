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



def get_cox_data():
    durations = pd.read_csv("oslo2/cell_distances.csv")

    n_cells = durations["start_cell"].max() + 1

    mu  = [[[] for end_cell in range(n_cells)] for start_cell in range(n_cells)]
    phi = [[[] for end_cell in range(n_cells)] for start_cell in range(n_cells)]

    # mu/phi format: [start_cell][end_cell][phase] => mu/phi
    for i, row in durations.iterrows():
        start_cell = int(row["start_cell"])
        end_cell   = int(row["end_cell"])

        if row["n"] == 1:
            mu[start_cell][end_cell]  = [float(row["lambda"])]
            phi[start_cell][end_cell] = [1.0]
        elif row["n"] == 2:
            mu[start_cell][end_cell]  = [float(row["mu1"]), float(row["mu2"])]
            phi[start_cell][end_cell] = [float(row["phi1"]), 1.0]
        else:
            erlang_phases = int(row["n"])
            lam = float(row["lambda"])
            mu[start_cell][end_cell]  = [lam for i in range(erlang_phases)]
            phi[start_cell][end_cell] = [0.0 for i in range(erlang_phases-1)] + [1.0]

    return [mu, phi]

def get_demands(cell_to_station, station_to_cell):
    in_demands_frame = pd.read_csv("oslo2/in_demands.csv")
    out_demands_frame = pd.read_csv("oslo2/out_demands.csv")
    in_probabilities_frame = pd.read_csv("oslo2/in_probs.csv")

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

        in_demands[end_cell][start_cell][end_station] = float(row["prob"])



    # start_cell => station => end_cell => demand
    out_demands = [[[0 for end_cell in range(n_cells)] for station in cell_to_station[start_cell]] for start_cell in range(n_cells)]

    for i, row in out_demands_frame.iterrows():
        start_station = int(row["start"])
        start_cell = station_to_cell[start_station]
        end_cell = int(row["end_cell"])
        start_station = cell_to_station[start_cell].index(start_station)
        

        out_demands[start_cell][start_station][end_cell] = float(row["hourly_demand"])



    return [in_demands, in_probabilities, out_demands]

def get_starting_bps():
    initial_loads = pd.read_csv("oslo2/initial_loads.csv")
    return list(initial_loads["start_level"])

def run_discrete(model_data, traj_cells, ode_method, step_size):
    print("Running Discrete-Step Submodeling")

    tic = time.perf_counter()

    n_entries = model_data.n_cells**2 + model_data.n_stations

    x_arr = np.array([])

    delay_phase_ct = [sum([len(x) for x in mu[i]]) for i in range(model_data.n_cells)] # number of phases in the process that starts at i
    in_phase_ct    = [sum([len(x[end_cell]) for x in mu]) for end_cell in range(model_data.n_cells)] # number of phases in the process that starts at i
    
    trajectories = [[0 for j in range(in_phase_ct[end_cell])] for end_cell in range(model_data.n_cells)]

    current_vector = [
        [0.0 for i in range(delay_phase_ct[cell_idx])] +
        [float(x) for i, x in enumerate(starting_bps) if i in list(cell_to_station[cell_idx])]
            for cell_idx in range(model_data.n_cells)
    ]
    station_vals = [[] for i in range(model_data.n_stations)]

    time_points = []

    t = 0

    
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
            if cell_idx % 10 == 0:
                print(f"cell: {cell_idx}")
            traj_cells[cell_idx].set_trajectories(trajectories[cell_idx])

            x_t = spi.solve_ivp(traj_cells[cell_idx].dxdt_const, [t, t+step_size], current_vector[cell_idx], 
                                    t_eval = sub_time_points,
                                    method=ode_method, atol=ATOL)


            x_iter = np.concatenate([x_iter, x_t.y])
            
            for i, station_id in enumerate(traj_cells[cell_idx].stations):
                sy_idx = delay_phase_ct[cell_idx]+i
                current_vector[cell_idx][sy_idx] = float(x_t.y[sy_idx, -1])

            #FIX: trajectory updates should be to the next cell
            for next_cell in range(model_data.n_cells):
                for phase in range(len(mu[cell_idx][next_cell])):
                    phase_qty_idx = traj_cells[cell_idx].x_idx[next_cell] + phase
                    last_val = float(x_t.y[phase_qty_idx, -1])
                    current_vector[cell_idx][phase_qty_idx] = last_val

                    for next_cell in range(model_data.n_cells):
                        new_trajectories[next_cell][traj_cells[cell_idx].x_in_idx[cell_idx] + phase] = last_val

        if x_arr.size == 0:
            x_arr = x_iter
        else:
            x_arr = np.concatenate([x_arr, x_iter], axis=1)

        trajectories = new_trajectories

        t += step_size
    toc = time.perf_counter()
    print(f"Discrete-Step Submodeling finished, time: {toc-tic}")
    return [time_points, x_arr, toc-tic]

class ModelData:
    def __init__(self, n_stations, n_cells, starting_bps, mu):
        self.n_stations = n_stations
        self.n_cells = n_cells
        self.starting_bps = starting_bps, 
        self.mu = mu
        self.time_end = 4
        self.steps_per_dt = 100

if __name__ == "__main__":
    # TO DO:
    #    [ ] accurate bps
    #    [ ] fix Delta T to something reasonable
    #    [ ] rerun ipynb
    #    [ ] why is mu[0] of length 291 rather than 58??

    # load data

    # stations
    stations = pd.read_csv("oslo2/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = stations["cell"].max() + 1
    n_stations = stations["index"].max() + 1
    cell_to_station = [[] for i in range(n_cells)]
    station_to_cell = [0 for i in range(n_stations)]
    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])
    
    # demands
    in_demands, in_probabilities, out_demands = get_demands(cell_to_station, station_to_cell)

    # phase processes
    mu, phi = get_cox_data()

    # build StrictTrajCellCox for each station cell
    traj_cells = [
        spatial_decomp_strict.StrictTrajCellCox(cell_idx, cell_to_station[cell_idx], 
            mu, phi, in_demands[cell_idx], in_probabilities[cell_idx], 
            out_demands[cell_idx])

            for cell_idx in range(n_cells)
    ]

    # run model
    starting_bps = get_starting_bps()
    model_data = ModelData(n_stations, n_cells, starting_bps, mu)
    time_points, x_arr, time = run_discrete(model_data, traj_cells, "RK45", 0.1)

    print(x_arr[:, -1])