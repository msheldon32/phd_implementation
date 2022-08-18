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

class TrajCell:
    def __init__(self, cell_idx, station_to_cell, cell_to_station, stations, durations, demands):
        self.cell_idx = cell_idx
        self.n_cells = len(cell_to_station)
        self.n_stations = len(durations)


        self.s_in_cell = len(stations)
        self.stations = stations

        self.durations = durations
        self.demands = demands

        self.station_to_cell = station_to_cell
        self.cell_to_station = cell_to_station
    
        self.trajectories = []

        self.total_rate = [0 for i in range(self.n_stations)]

        for j, station_idx in enumerate(self.stations):
            for dest_station in range(self.n_stations):
                self.total_rate[j] += self.demands[station_idx][dest_station]
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories
    
    def get_delay_idx(self, i_idx, d_idx):
        # i_idx: index of the source station in self.stations (e.g. the internal index)
        # d_idx: overall index of the destination station
        return (i_idx*self.n_stations) + d_idx

    def get_station_idx(self, i_idx):
        # i_idx: index of the source station in self.stations (e.g. the internal index)
        return (self.n_stations * self.s_in_cell) + i_idx 

    def dxdt(self, t, x):
        deriv = [0 for i in range((self.n_stations*self.s_in_cell)+self.s_in_cell)]
        
        # inline re-implementation for speed
        get_delay_idx = lambda i_idx, d_idx: (i_idx*self.n_stations) + d_idx
        get_station_idx = lambda i_idx: (self.n_stations * self.s_in_cell) + i_idx

        # derive output delays
        for j, src_station in enumerate(self.stations):
            for dst_station in range(self.n_stations):
                    d_idx = get_delay_idx(j, dst_station)
                    src_idx = (self.n_stations*self.s_in_cell) + j
                    deriv[d_idx] += self.demands[src_station][dst_station]*min(x[src_idx], 1)
                    deriv[d_idx] -= (1/self.durations[src_station][dst_station])*x[d_idx]

        
        # derive station levels
        for j, dst_station in enumerate(self.stations):
            d_idx = (self.n_stations*self.s_in_cell)+j

            for src_station in range(self.n_stations):
                rate = 1/self.durations[src_station][dst_station]
                if src_station in self.cell_to_station[self.cell_idx]:
                    # the source is internal to the cell, so we can solve it with x
                    corres_idx = self.stations.index(src_station)
                    src_d_idx = get_delay_idx(corres_idx, dst_station)
                    deriv[d_idx] += rate*x[src_d_idx]
                else:
                    deriv[d_idx] += rate*self.trajectories[src_station][dst_station](t)
            
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