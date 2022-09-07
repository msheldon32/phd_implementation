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
    
        self.trajectories = []

        self.tstep = 0

    def set_timestep(self, tstep):
        self.tstep = tstep
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories
    
    def dxdt(self, t, x):
        deriv = [0 for i in range(self.n_cells+self.s_in_cell)]
        
        for i in range(self.n_cells):
            for j, station_idx in enumerate(self.stations):
                #d_idx = j + self.n_cells
                deriv[i] += self.out_demands[i][i]*min(x[j + self.n_cells],1)
            deriv[i] -= (1/self.durations[self.cell_idx][i])*x[i]
            
        for j, station_idx in enumerate(self.stations):
            #d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[j])

            
            for i in range(self.n_cells):                    
                rate = self.in_probabilities[i][j]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[j + self.n_cells] += rate*x[self.cell_idx]
                else:
                    deriv[j + self.n_cells] += rate*self.trajectories[i](t)

            deriv[j + self.n_cells] -= station_demand*min(x[j + self.n_cells],1)

        return deriv
    
    def get_idx(self):
        out_list = []

        for dst_cell in range(self.n_cells):
                out_list.append(self.n_cells*self.cell_idx + dst_cell)

        out_list += [(self.n_cells**2) + i for i in self.stations]
        return out_list
    
    def dxdt_const(self, t, x):
        deriv = [0 for i in range(self.n_cells+self.s_in_cell)]
        
        for i in range(self.n_cells):
            for j, station_idx in enumerate(self.stations):
                #d_idx = j + self.n_cells
                deriv[i] += self.out_demands[j][i]*min(x[j + self.n_cells],1)
            deriv[i] -= (1/self.durations[self.cell_idx][i])*x[i]
            
        for j, station_idx in enumerate(self.stations):
            #d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[j])

            
            for i in range(self.n_cells):                    
                rate = self.in_probabilities[i][j]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[j + self.n_cells] += rate*x[self.cell_idx]
                else:
                    deriv[j + self.n_cells] += rate*self.trajectories[i]
                    
            deriv[j + self.n_cells] -= station_demand*min(x[j + self.n_cells],1)

        return deriv
    
    def dxdt_array(self, t, x):
        deriv = [0 for i in range(self.n_cells+self.s_in_cell)]
        
        for i in range(self.n_cells):
            for j, station_idx in enumerate(self.stations):
                assert j < self.s_in_cell
                deriv[i] += self.out_demands[j][i]*min(x[j + self.n_cells],1)
            deriv[i] -= (1/self.durations[self.cell_idx][i])*x[i]
            
        for j, station_idx in enumerate(self.stations):
            #d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[j])

            
            for i in range(self.n_cells):                    
                rate = self.in_probabilities[i][j]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[j + self.n_cells] += rate*x[self.cell_idx]
                else:
                    deriv[j + self.n_cells] += rate*self.trajectories[i][int(t//self.tstep)]
                    
            deriv[j + self.n_cells] -= station_demand*min(x[j + self.n_cells],1)
        assert(len(deriv) == len(x))
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


class StrictTrajCellCox:
    def __init__(self, cell_idx, stations, mu, phi, in_demands, in_probabilities, out_demands):
        self.n_cells = len(in_demands)
        self.stations = stations
        self.s_in_cell = len(stations)

        self.cell_idx = cell_idx
        
        self.mu = mu
        self.phi = phi

        self.n_phases = [len(self.mu[i][self.cell_idx]) for i in range(self.n_cells)]

        self.x_idx = [0 for i in range(self.n_cells)]
        self.station_offset = 0

        for end_cell in range(self.n_cells):
            self.x_idx[end_cell] = self.x_idx
            self.station_offset += self.n_phases[end_cell]

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
        deriv = [0 for i in range(self.station_offset+self.s_in_cell)]
        
        for end_cell in range(self.n_cells):
            d_idx = self.x_idx[end_cell]

            for j, station_idx in enumerate(self.stations):
                s_idx = j + self.n_cells
                deriv[s_idx] += self.out_demands[j][end_cell]*min(x[s_idx],1)
            
            for phase in range(self.n_phases[end_cell]):
                deriv[d_idx + phase] -= self.mu[self.cell_idx][end_cell] * x[d_idx+phase]

        ################
            
        for j, station_idx in enumerate(self.stations):
            #d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[j])

            for i in range(self.n_cells):                    
                rate = self.in_probabilities[i][j]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[j + self.n_cells] += rate*x[self.cell_idx]
                else:
                    deriv[j + self.n_cells] += rate*self.trajectories[i](t)
            deriv[j + self.n_cells] -= station_demand*min(x[j + self.n_cells],1)

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