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

DATA_FOLDER = "oslo_data_4/"
HOURLY_CLUSTER = [hr for hr in range(0,24)] # map hour->time cluster

def get_oslo_data(time_clusters = HOURLY_CLUSTER):
    in_cell_demands  = pd.read_csv(DATA_FOLDER + "in_cell_demands.csv", index_col=0)
    out_cell_demands = pd.read_csv(DATA_FOLDER + "out_cell_demands.csv", index_col=0)
    cell_durations   = pd.read_csv(DATA_FOLDER + "cell_durations.csv", index_col=0)
    clusters         = pd.read_csv(DATA_FOLDER +"station_clusters.csv").rename({"Unnamed: 0": "station"},axis=1)

    n_time_clusters = len(set(time_clusters))
    n_stations = clusters["station"].max()+1
    n_cells = clusters["cell"].max()+1

    # useful maps and functions
    cluster_map = {
        time_cluster: set() for time_cluster in range(n_time_clusters)
    }
    for hr, time_cluster in enumerate(time_clusters):
        cluster_map[time_cluster].add(hr)
    
    cell_map = [0 for i in range(n_stations)]

    for i, row in clusters.iterrows():
        cell_map[int(row["station"])] = int(row["cell"])

    # [time][start_cell][end_cell]: duration to transit from start to end
    duration_array = [[
        [[0,0] for end_cell in range(n_cells)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]


    overall_durations = [[
        [0,0] for end_cell in range(n_cells)]
            for start_cell in range(n_cells)]


    for i, row in cell_durations.iterrows():
        time_cluster = time_clusters[int(row["hour"])]
        duration_array[time_cluster][int(row["start_cell"])][int(row["end_cell"])][0] += float(row["duration_hr"])
        duration_array[time_cluster][int(row["start_cell"])][int(row["end_cell"])][1] += 1
        overall_durations[int(row["start_cell"])][int(row["end_cell"])][0] += float(row["duration_hr"])
        overall_durations[int(row["start_cell"])][int(row["end_cell"])][1] += 1
    
    for time_cluster in range(n_time_clusters):
        for start_cell in range(n_cells):
            for end_cell in range(n_cells):
                tup = duration_array[time_cluster][start_cell][end_cell]
                if tup[1] != 0:
                    duration_array[time_cluster][start_cell][end_cell] = tup[0]/tup[1]
                else:
                    # interpolate: assume the average time over all hours
                    overall_tup = overall_durations[start_cell][end_cell]
                    duration_array[time_cluster][start_cell][end_cell] = overall_tup[0]/overall_tup[1]
        
    
    # [time][start_cell][station]: total demand from start_cell to station in time
    in_demands = [[[0
            for station in range(n_stations)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]
    
    # [time][start_cell][end_cell]: total demand from start_cell to end_cell
    in_demands_cell = [[[0
            for end_cell in range(n_cells)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]

    for i, row in in_cell_demands.iterrows():
        time_cluster = time_clusters[int(row["hour"])]
        station = int(row["end"])
        end_cell = cell_map[station]
        start_cell = int(row["start_cell"])
        in_demands[time_cluster][start_cell][station]       += float(row["hourly_rate"])
        in_demands_cell[time_cluster][start_cell][end_cell] += float(row["hourly_rate"])

    
    # [time][start_cell][station]: probability to transition to station
    in_probabilities = [[[
                in_demands[time_cluster][start_cell][station] / in_demands_cell[time_cluster][start_cell][cell_map[station]]
                    if in_demands_cell[time_cluster][start_cell][cell_map[station]] != 0 else 0
            for station in range(n_stations)]
            for start_cell in range(n_cells)]
            for time_cluster in range(n_time_clusters)]
    
    # [time][station][end_cell]: total demand from start_cell to station in time
    out_demands = [[[
                0
            for end_cell in range(n_cells)]
            for station in range(n_stations)]
            for time_cluster in range(n_time_clusters)]
    
    for i, row in out_cell_demands.iterrows():
        time_cluster = time_clusters[int(row["hour"])]
        station  = int(row["start"])
        end_cell = int(row["end_cell"])
        out_demands[time_cluster][station][end_cell] += float(row["hourly_rate"])
    
    return [clusters, duration_array, in_demands, in_demands_cell, in_probabilities, out_demands]
