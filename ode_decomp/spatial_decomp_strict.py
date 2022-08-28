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

class StrictTrajCell:
    def __init__(self, cell_idx, stations, durations, in_demands, in_probabilities, out_demands):
        self.n_cells = len(durations)
        self.stations = stations
        self.s_in_cell = len(stations)

        self.cell_idx = cell_idx
        self.durations = durations
        self.in_probabilities = in_probabilities
        self.out_demands = out_demands

        if in_demands != None:
            total_rate_in  = 0
            total_rate_out = 0
            for start_cell in range(self.n_cells):
                if start_cell == cell_idx:
                    continue
                total_rate_in += in_demands[start_cell][cell_idx]

            for end_cell in range(self.n_cells):
                if start_cell == cell_idx:
                    continue
                total_rate_out += in_demands[cell_idx][end_cell]
            if total_rate_out == 0:
                self.io_rate = 1
            else:
                self.io_rate = total_rate_in/total_rate_out
            self.first_iteration = True
        else:
            self.io_rate = 1
            self.first_iteration = False
    
        self.trajectories = []
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories
    
    def dxdt(self, t, x):
        deriv = [0 for i in range(self.n_cells+self.s_in_cell)]
        
        for i in range(self.n_cells):
            for j, station_idx in enumerate(self.stations):
                d_idx = j + self.n_cells
                deriv[i] += self.out_demands[station_idx][i]*min(x[d_idx],1)
            deriv[i] -= (1/self.durations[self.cell_idx][i])*x[i]
            
        for j, station_idx in enumerate(self.stations):
            d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[station_idx])

            
            for i in range(self.n_cells):                    
                rate = self.in_probabilities[i][station_idx]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[d_idx] += rate*x[self.cell_idx]
                else:
                    deriv[d_idx] += rate*self.trajectories[i](t)
            deriv[d_idx] -= station_demand*min(x[d_idx],1)

            if self.first_iteration:
                demand_feedback = station_demand
                demand_feedback -= self.out_demands[station_idx][self.cell_idx]
                demand_feedback *= self.io_rate
                deriv[d_idx] += demand_feedback*min(x[d_idx],1)

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