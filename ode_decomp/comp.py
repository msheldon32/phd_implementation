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

class CompModel:
    def __init__(self, stations, durations, demands):
        self.cell_idx = cell_idx
        self.n_stations = len(durations)

        self.stations = stations

        self.durations = durations
        self.demands = demands

        self.total_rate = [0 for i in range(self.n_stations)]

        for j, station_idx in enumerate(self.stations):
            for dest_station in range(self.n_stations):
                self.total_rate[j] += self.demands[station_idx][dest_station]
    
    def get_delay_idx(self, s_idx, d_idx):
        # s_idx: overall index of the source station
        # d_idx: overall index of the destination station
        return (s_idx*self.n_stations) + d_idx

    def get_station_idx(self, s_idx):
        # s_idx: overall index of the source station
        # d_idx: overall index of the destination station
        return (self.n_stations * self.self.n_stations) + s_idx 

    def dxdt(self, t, x):
        deriv = [0 for i in range((self.n_stations*self.n_stations)+self.n_stations)]
        
        # inline re-implementation for speed
        get_delay_idx = lambda s_idx, d_idx: (s_idx*self.n_stations) + d_idx
        get_station_idx = lambda s_idx, d_idx: (self.n_stations * self.self.n_stations) + s_idx 

        # derive delay levels
        for src_station in range(self.n_stations):
            for dst_station in range(self.n_stations):
                    d_idx = get_delay_idx(j, dst_station)
                    src_idx = get_station_idx(src_station)
                    deriv[d_idx] += self.demands[src_station][dst_station]*min(x[src_idx], 1)
                    deriv[d_idx] -= (1/self.durations[src_station][dst_station])*x[d_idx]
        
        # derive station levels
        for dst_station in range(self.n_stations):
            dst_idx = get_station_idx(dst_station)

            for source_station in range(self.n_stations):
                rate = 1/self.durations[source_station][dest_station]
                if source_station in self.cell_to_station[self.cell_idx]:
                    # the source is internal to the cell, so we can solve it with x
                    corres_idx = self.stations.index(source_station)
                    x_idx = (self.n_stations*corres_idx) + dest_station
                    deriv[dst_idx] += rate*x[x_idx]
                else:
                    deriv[dst_idx] += rate*self.trajectories[source_station][dest_station](t)
            
            deriv[dst_idx] -= self.total_rate[j]*min(x[dst_idx],1)

        return deriv

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