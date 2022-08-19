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

import sklearn.cluster

import spatial_decomp_station
import comp
from oslo_data import *
import large_fluid


def generate_uniform(interval):
    lo, hi = interval
    r = hi-lo
    return random.random()*r + lo

def generate_gaussian(moments):
    # moments: [mean, std. dev]
    return np.random.normal(moments[0], moments[1])

def euclidian_distance(a, b):
    sum_squares = 0

    for aval, bval in zip(a,b):
        sum_squares += (aval-bval)**2
    
    return sum_squares ** 0.5

class ExperimentModel:
    def __init__(self, stations, durations, demands, seed):
        self.n_stations = len(stations)

        self.stations  = stations
        self.durations = durations
        self.demands   = demands

        self.n_cells = 1
        self.station_to_cell = [0 for i in range(self.n_stations)]
        self.cell_to_station = [set([i for i in range(self.n_stations)])]

        self.seed = seed
    
    def generate_cells(self, stations_per_cell):
        self.n_cells = round(self.n_stations/stations_per_cell)

        self.cell_to_station = [set() for i in range(self.n_cells)]

        clusterer = sklearn.cluster.KMeans(n_clusters=self.n_cells, random_state=self.seed)
        clusterer.fit(self.stations)

        self.station_to_cell = [x for x in clusterer.predict(self.stations)]
        
        for i, cell in enumerate(self.station_to_cell):
            self.cell_to_station[cell].add(i)
            

class ExperimentConfig:
    def __init__(self):
        # meta
        self.repetitions_per_point = 300

        # parameters
        self.ode_methods             = ["BDF", "RK45"]
        self.stations_per_cell       = [5, 10, 15, 20]
        self.delta_t_ratio           = [0.5, 0.1, 0.05, 0.01, 0.005] # setting delta T based on (x/[average rate])

        # random configuration
        self.n_station_range         = [50, 100]
        self.x_location_range        = [0, 1]
        self.y_location_range        = [0, 1]
        self.station_demand_range    = [0, 0.5]
        self.noise_moments_distance  = [0, 0.2]

        # constants
        self.n_iterations = 25
        self.time_end     = 24

        self.seed = 1996

    
    def generate_stations(self):
        n_stations = round(generate_uniform(self.n_station_range))

        station_list = []

        for station_no in range(n_stations):
            x_loc = generate_uniform(self.x_location_range)
            y_loc = generate_uniform(self.y_location_range)
            station_list.append([x_loc, y_loc])
        
        return station_list
    
    def generate_durations(self, stations):
        n_stations = len(stations)
        durations = [[0 for i in range(n_stations)] for j in range(n_stations)]

        for i, src_stn in enumerate(stations):
            for j, dst_stn in enumerate(stations):
                distance = euclidian_distance(src_stn, dst_stn)
                eps = generate_gaussian(self.noise_moments_distance)
                durations[i][j] = distance + eps
        return durations

    def generate_demands(self, stations):
        n_stations = len(stations)
        demands = [[0 for i in range(n_stations)] for j in range(n_stations)]

        for i, src_stn in enumerate(stations):
            for j, dst_stn in enumerate(stations):
                demands[i][j] = generate_uniform(self.station_demand_range)

    def generate_model(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed += 1

        stations = self.generate_stations()
        durations = self.generate_durations(stations)
        demands = self.generate_demands(stations)

        return ExperimentModel(stations, durations, demands, self.seed)

class Experiment:
    def __init__(self, configuration):
        self.configuration = configuration

        self.output_folder = ""
    
    def run_full(self, model):
        pass
    
    def run_iteration(self, model):
        pass
    
    def run_discrete(self, model):
        pass

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(self.output_folder + "output.csv", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["solution_method", "spatial_simplification", "ode_method", "stations_per_cell", 
                             "iteration", "delta_t", "station_no", "delta"])
        try:
            for repetition in range(self.configuration.repetitions_per_point):
                print(f"Repetition: {repetition}")
                model = self.configuration.generate_model()

                for ode_method in self.configuration.ode_methods:
                    full_res = self.run_full(model)

                    for stations_per_cell in self.configuration.stations_per_cell:
                        model.generate_cells(stations_per_cell)

                        iter_res = self.run_iteration(model)

                        for delta_t_ratio in self.configuration.delta_t_ratio:
                            disc_res = self.run_discrete(model)
        except:
            os.remove(self.output_folder + "output.csv")
            raise