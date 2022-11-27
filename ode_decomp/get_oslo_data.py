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
DEMAND_INFLATION = 1 # FOR TESTING/EXPERIMENTS ONLY
TEST_PARAM = False

data_folder = "oslo_data_3_big"

def get_cox_data(hour, n_cells):
    durations = pd.read_csv(f"{data_folder}/cell_distances.csv")
    default_durations = pd.read_csv(f"{data_folder}/default_cell_distances.csv")

    durations = durations[durations["hour"] == hour]

    mu  = [[[1] for end_cell in range(n_cells)] for start_cell in range(n_cells)]
    phi = [[[1.0] for end_cell in range(n_cells)] for start_cell in range(n_cells)]

    # mu/phi format: [start_cell][end_cell][phase] => mu/phi

    # first run through general durations then hour-specific ones
    for i, row in default_durations.iterrows():
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

    n_cells = in_demands_frame["start_cell"].max() + 1

    in_demands_frame = in_demands_frame[in_demands_frame["hour"] == hour]
    out_demands_frame = out_demands_frame[out_demands_frame["hour"] == hour]
    in_probabilities_frame = in_probabilities_frame[in_probabilities_frame["hour"] == hour]

    # end_cell => start_cell => station => demand
    in_demands = [[[0 for station in cell_to_station[end_cell]] for start_cell in range(n_cells)] for end_cell in range(n_cells)]

    for i, row in in_demands_frame.iterrows():
        start_cell = int(row["start_cell"])
        end_station = int(row["end"])
        end_cell = station_to_cell[end_station]
        end_station = cell_to_station[end_cell].index(end_station)
        
        in_demands[end_cell][start_cell][end_station] = float(row["hourly_demand"]) * DEMAND_INFLATION


    # end_cell => start_cell => station => probability
    in_probabilities = [[[0 for station in cell_to_station[end_cell]] for start_cell in range(n_cells)] 
                    for end_cell in range(n_cells)]

    for i, row in in_probabilities_frame.iterrows():
        start_cell = int(row["start_cell"])
        end_station = int(row["end"])
        end_cell = station_to_cell[end_station]
        end_station = cell_to_station[end_cell].index(end_station)

        in_probabilities[end_cell][start_cell][end_station] = float(row["prob"])
    
    for start_cell in range(n_cells):
        for end_cell in range(n_cells):
            if sum(in_probabilities[end_cell][start_cell]) == 0:
                n_stns = len(list(cell_to_station[end_cell]))
                in_probabilities[end_cell][start_cell] = [1/n_stns for i in range(n_stns)]
    
    


    # start_cell => station => end_cell => demand
    out_demands = [[[0 for end_cell in range(n_cells)] for station in cell_to_station[start_cell]] for start_cell in range(n_cells)]

    for i, row in out_demands_frame.iterrows():
        start_station = int(row["start"])
        start_cell = station_to_cell[start_station]
        end_cell = int(row["end_cell"])
        start_station = cell_to_station[start_cell].index(start_station)
        

        out_demands[start_cell][start_station][end_cell] = float(row["hourly_demand"]) * DEMAND_INFLATION

        if TEST_PARAM:
            out_demands[start_cell][start_station][end_cell] = (random.random()**2) *0.2* DEMAND_INFLATION



    return [in_demands, in_probabilities, out_demands]

def get_average_probs(start_hour, end_hour, cell_to_station, station_to_cell):
    in_probs = []
    for hr in range(start_hour, end_hour):
        in_probs.append(get_demands(hr, cell_to_station, station_to_cell)[1])
    

    n_cells = len(cell_to_station)

    in_probs_out = []

    for end_cell in range(len(in_probs[0])):
        in_probs_out.append([])
        for start_cell in range(len(in_probs[0][end_cell])):
            in_probs_out[end_cell].append([])
            for stn in range(len(in_probs[0][end_cell][start_cell])):
                in_probs_out[end_cell][start_cell].append(0)
                for hr_idx in range(0,end_hour-start_hour):
                    in_probs_out[end_cell][start_cell][stn] += in_probs[hr_idx][end_cell][start_cell][stn]
                in_probs_out[end_cell][start_cell][stn] /= (end_hour-start_hour)
    return in_probs_out

def get_starting_bps():
    initial_loads = pd.read_csv(f"{data_folder}/initial_loads.csv")
    return list(initial_loads["start_level"])