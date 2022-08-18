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
    def __init__(self, durations, demands):
        self.n_stations = len(durations)

        self.durations = durations
        self.demands = demands

        self.total_rate = [0 for i in range(self.n_stations)]

        for src_station in range(self.n_stations):
            for dst_station in range(self.n_stations):
                self.total_rate[src_station] += self.demands[src_station][dst_station]
    
    def get_delay_idx(self, src_idx, dst_idx):
        # src_idx: overall index of the source station
        # dst_idx: overall index of the destination station
        return (src_idx*self.n_stations) + dst_idx

    def get_station_idx(self, stn_idx):
        # stn_idx: overall index of the source station
        return (self.n_stations ** 2) + stn_idx 

    def dxdt(self, t, x):
        deriv = [0 for i in range((self.n_stations ** 2)+self.n_stations)]
        
        # inline re-implementation for speed
        get_delay_idx = lambda src_idx, dst_idx: (src_idx*self.n_stations) + dst_idx
        get_station_idx = lambda stn_idx: (self.n_stations ** 2) + stn_idx 

        # derive delay levels
        for src_station in range(self.n_stations):
            for dst_station in range(self.n_stations):
                    src_idx = get_station_idx(src_station)
                    del_idx = get_delay_idx(src_station, dst_station)
                    deriv[del_idx] += self.demands[src_station][dst_station]*min(x[src_idx], 1)
                    deriv[del_idx] -= (1/self.durations[src_station][dst_station])*x[del_idx]
        
        # derive station levels
        for cur_station in range(self.n_stations):
            cur_idx = get_station_idx(cur_station)

            for src_station in range(self.n_stations):
                rate = 1/self.durations[src_station][cur_station]
                src_idx = get_delay_idx(src_station, cur_station)
                deriv[cur_idx] += rate*x[src_idx]
            
            deriv[cur_idx] -= self.total_rate[cur_station]*min(x[cur_idx],1)

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