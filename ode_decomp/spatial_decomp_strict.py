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
import copy

ELASTICITY = 1.0

def get_loss_ptg(mean, capacity):
    rho = 0 # find this

    return (rho**(capacity))*(1-rho)/(1-(rho**(capacity*1)))

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
            #print("---------------")
            #print(self.durations[self.cell_idx][i])
            #print(1/self.durations[self.cell_idx][i])
            
        for j, station_idx in enumerate(self.stations):
            #d_idx = j + self.n_cells

            station_demand = sum(self.out_demands[j])

            
            for i in range(self.n_cells):
                rate = self.in_probabilities[i][j]*(1/self.durations[i][self.cell_idx])
                if i == self.cell_idx:
                    deriv[j + self.n_cells] += rate*x[self.cell_idx]
                else:
                    deriv[j + self.n_cells] += rate*self.trajectories[i][int(t//self.tstep)]
            #print(deriv[j+self.n_cells])
            deriv[j + self.n_cells] -= station_demand*min(x[j + self.n_cells],1)
        #assert(len(deriv) == len(x))
        #raise Exception()
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

        self.n_phases_in = [len(self.mu[i][self.cell_idx]) for i in range(self.n_cells)]
        self.n_phases_out = [len(self.mu[self.cell_idx][i]) for i in range(self.n_cells)]

        self.x_idx = [0 for i in range(self.n_cells)]
        self.x_in_idx = [0 for i in range(self.n_cells)]
        self.station_offset = 0
        self.in_offset = 0

        self.prices = [1 for i in range(self.s_in_cell)]

        self.in_demands = in_demands

        for end_cell in range(self.n_cells):
            self.x_idx[end_cell] = self.station_offset
            self.station_offset += self.n_phases_out[end_cell]

            self.x_in_idx[end_cell] = self.in_offset
            self.in_offset += self.n_phases_in[end_cell]

        self.in_probabilities = in_probabilities
        self.out_demands = out_demands
    
        self.trajectories = []

        self.tstep = 0

        self.cache = {}
    
    def cache(self):
        self.cache["out_demands"] = self.out_demands
        self.cache["prices"] = self.prices
    
    def uncache(self):
        self.out_demands = self.cache["out_demands"]
        self.prices = self.cache["prices"]
    
    def set_subsidy(self, station):
        # halve price, multiple
        for end_cell in range(self.n_cells):
            self.out_demands[station][end_cell] *= 1.5
            self.prices[station] /= 2
    
    def remove_subsidy(self, station):
        for end_cell in range(self.n_cells):
            self.out_demands[station][end_cell] /= 1.5
            self.prices[station] *= 2

    def set_timestep(self, tstep):
        self.tstep = tstep
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories
    
    def dxdt(self, t, x):
        deriv = [0 for i in range(self.station_offset+self.s_in_cell)]
        
        for end_cell in range(self.n_cells):
            d_idx = self.x_idx[end_cell]

            for j, station_idx in enumerate(self.stations):
                s_idx = j + self.station_offset
                deriv[d_idx] += self.out_demands[j][end_cell]*min(x[s_idx],1)
            
            for phase in range(self.n_phases_out[end_cell]):
                deriv[d_idx + phase] -= self.mu[self.cell_idx][end_cell][phase] * x[d_idx+phase]
                
                if phase < self.n_phases_out[end_cell]-1:
                    deriv[d_idx + phase + 1] += self.mu[self.cell_idx][end_cell][phase] * (1-self.phi[self.cell_idx][end_cell][phase]) * x[d_idx+phase]
        
        for j, station_idx in enumerate(self.stations):
            station_demand = sum(self.out_demands[j])

            deriv[j + self.station_offset] -= station_demand*min(x[j + self.station_offset],1)

            for start_cell in range(self.n_cells):
                for phase in range(self.n_phases_in[start_cell]):
                    rate = self.in_probabilities[start_cell][j]*self.mu[start_cell][self.cell_idx][phase]*self.phi[start_cell][self.cell_idx][phase]
                    if start_cell == self.cell_idx:
                        deriv[j + self.station_offset] += rate*x[self.x_idx[self.cell_idx]+phase]
                    else:
                        deriv[j + self.station_offset] += rate*self.trajectories[self.x_in_idx[start_cell]+phase](t)

        return deriv
    
    def x_size(self):
        return self.station_offset+self.s_in_cell
    
    def get_idx(self):
        raise Exception("not implemented..")
        # need to update this to add in coxian-distributed delays
        out_list = []

        for dst_cell in range(self.n_cells):
                out_list.append(self.n_cells*self.cell_idx + dst_cell)

        out_list += [(self.n_cells**2) + i for i in self.stations]
        return out_list
    
    def set_prices(self, prices):
        self.prices = prices
    
    def dxdt_const(self, t, x):
        deriv = [0 for i in range(self.station_offset+self.s_in_cell)]
        
        for end_cell in range(self.n_cells):
            d_idx = self.x_idx[end_cell]

            for j, station_idx in enumerate(self.stations):
                deriv[d_idx] += self.out_demands[j][end_cell]*min(x[j + self.station_offset],1)
            
            for phase in range(self.n_phases_out[end_cell]):
                deriv[d_idx + phase] -= self.mu[self.cell_idx][end_cell][phase] * x[d_idx+phase]
                
                if phase < self.n_phases_out[end_cell]-1:
                    deriv[d_idx + phase + 1] += self.mu[self.cell_idx][end_cell][phase] * (1-self.phi[self.cell_idx][end_cell][phase]) * x[d_idx+phase]

        for j, station_idx in enumerate(self.stations):
            station_demand = sum(self.out_demands[j])

            deriv[j + self.station_offset] -= station_demand*min(x[j + self.station_offset],1)

            for start_cell in range(self.n_cells):
                # hook
                #print(sum(self.in_probabilities[start_cell]))
                #assert 0.99 < sum(self.in_probabilities[start_cell]) < 1.01

                for phase in range(self.n_phases_in[start_cell]):
                    rate = self.in_probabilities[start_cell][j]*self.mu[start_cell][self.cell_idx][phase]*self.phi[start_cell][self.cell_idx][phase]

                    if start_cell == self.cell_idx:
                        deriv[j + self.station_offset] += rate*x[self.x_idx[self.cell_idx]+phase]
                    else:
                        deriv[j + self.station_offset] += rate*self.trajectories[self.x_in_idx[start_cell]+phase]
                        
        return deriv
    
    def dxdt_array(self, t, x):
        deriv = [0 for i in range(self.station_offset+self.s_in_cell + 2)]

        reward = 0
        regret = 0
        
        for end_cell in range(self.n_cells):
            d_idx = self.x_idx[end_cell]

            for j, station_idx in enumerate(self.stations):
                s_idx = j + self.station_offset
                
                deriv[d_idx] += self.out_demands[j][end_cell]*min(x[s_idx],1)
            
            for phase in range(self.n_phases_out[end_cell]):
                deriv[d_idx + phase] -= self.mu[self.cell_idx][end_cell][phase] * x[d_idx+phase]
                
                if phase < self.n_phases_out[end_cell]-1:
                    deriv[d_idx + phase + 1] += self.mu[self.cell_idx][end_cell][phase] * (1-self.phi[self.cell_idx][end_cell][phase]) * x[d_idx+phase]
        
        for j, station_idx in enumerate(self.stations):
            station_demand = sum(self.out_demands[j])

            # integrate over prices and lost trips
            reward += self.prices[j]*station_demand*min(x[j + self.station_offset],1)
            regret += max(1-(station_demand*min(x[j + self.station_offset],1)),0)

            deriv[j + self.station_offset] -= station_demand*min(x[j + self.station_offset],1)

            for start_cell in range(self.n_cells):
                for phase in range(self.n_phases_in[start_cell]):
                    rate = self.in_probabilities[start_cell][j]*self.mu[start_cell][self.cell_idx][phase]*self.phi[start_cell][self.cell_idx][phase]
                    if start_cell == self.cell_idx:
                        deriv[j + self.station_offset] += rate*x[self.x_idx[self.cell_idx]+phase]
                    else:
                        deriv[j + self.station_offset] += rate*self.trajectories[self.x_in_idx[start_cell]+phase][int(t//self.tstep)]

        deriv[-2] = regret
        deriv[-1] = reward

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



class StrictTrajCellCoxControl:
    def __init__(self, cell_idx, stations, mu, phi, in_demands, in_probabilities, out_demands):
        """
            Note:
                denominate everything by hour - start_hour
        """
        self.n_cells = len(in_demands[0])
        self.n_hours = len(in_demands)
        self.stations = stations
        self.s_in_cell = len(stations)

        self.cell_idx = cell_idx
        
        self.mu = mu
        self.phi = phi

        self.n_phases_in = [len(self.mu[0][i][self.cell_idx]) for i in range(self.n_cells)]
        self.n_phases_out = [len(self.mu[0][self.cell_idx][i]) for i in range(self.n_cells)]

        self.x_idx = [0 for i in range(self.n_cells)]
        self.x_in_idx = [0 for i in range(self.n_cells)]
        self.station_offset = 0
        self.in_offset = 0

        self.prices = [1.0 for i in range(self.s_in_cell)]
        self.price = 1.0

        self.in_demands = in_demands

        for end_cell in range(self.n_cells):
            self.x_idx[end_cell] = self.station_offset
            self.station_offset += self.n_phases_out[end_cell]

            self.x_in_idx[end_cell] = self.in_offset
            self.in_offset += self.n_phases_in[end_cell]

        self.in_probabilities = in_probabilities
        self.out_demands = out_demands
    
        self.trajectories = []

        self.tstep = 0

        self.cache = {}
        self.setcache()
    
    def setcache(self):
        self.cache["out_demands"] = copy.deepcopy(self.out_demands)
        self.cache["prices"] = copy.deepcopy(self.prices)
        self.cache["price"] = copy.deepcopy(self.price)
    
    def uncache(self):
        self.out_demands = copy.deepcopy(self.cache["out_demands"])
        self.prices = copy.deepcopy(self.cache["prices"])
        self.price = copy.deepcopy(self.cache["price"])

    def set_price(self, price):
        price = max(min(price, 2),0) # bound price between 0, 2
        self.uncache()

        self.prices = [price for stn in range(self.s_in_cell)]
        self.price = price
        for hr in range(self.n_hours):
            for end_cell in range(self.n_cells):
                for stn in range(self.s_in_cell):
                    self.out_demands[hr][stn][end_cell] += self.out_demands[hr][stn][end_cell]*ELASTICITY*(1-self.price)

    
    def set_subsidy(self, station):
        # halve price, multiply demand
        self.prices[station] *= 0.75
        for hr in range(self.n_hours):
            for end_cell in range(self.n_cells):
                self.out_demands[hr][station][end_cell] *= 1.25
    
    def remove_subsidy(self, station):
        # halve price, multiply demand
        self.prices[station] /= 0.75
        for hr in range(self.n_hours):
            for end_cell in range(self.n_cells):
                self.out_demands[hr][station][end_cell] /= 1.25
    
    def subsidize_cell(self):
        for stn in range(self.s_in_cell):
            self.set_subsidy(stn)
    
    def unsubsidize_cell(self):
        for stn in range(self.s_in_cell):
            self.remove_subsidy(stn)

    def set_timestep(self, tstep):
        self.tstep = tstep
    
    def set_trajectories(self, trajectories):
        self.trajectories = trajectories
    
    def x_size(self):
        return self.station_offset+self.s_in_cell
    
    def get_idx(self):
        raise Exception("not implemented..")
        # need to update this to add in coxian-distributed delays
        out_list = []

        for dst_cell in range(self.n_cells):
                out_list.append(self.n_cells*self.cell_idx + dst_cell)

        out_list += [(self.n_cells**2) + i for i in self.stations]
        return out_list
    
    def set_prices(self, prices):
        self.prices = prices
        
    
    def dxdt_array(self, t, x):
        hr = math.floor(t)

        deriv = [0 for i in range(self.station_offset+self.s_in_cell+2)]

        #print(len(deriv))

        regret = 0
        reward = 0
        
        for end_cell in range(self.n_cells):
            d_idx = self.x_idx[end_cell]

            for j, station_idx in enumerate(self.stations):
                s_idx = j + self.station_offset
                
                deriv[d_idx] += self.out_demands[hr][j][end_cell]*min(x[s_idx],1)
            
            for phase in range(self.n_phases_out[end_cell]):
                deriv[d_idx + phase] -= self.mu[hr][self.cell_idx][end_cell][phase] * x[d_idx+phase]
                
                if phase < self.n_phases_out[end_cell]-1:
                    deriv[d_idx + phase + 1] += self.mu[hr][self.cell_idx][end_cell][phase] * (1-self.phi[hr][self.cell_idx][end_cell][phase]) * x[d_idx+phase]
        
        for j, station_idx in enumerate(self.stations):
            station_demand = sum(self.out_demands[hr][j])

            deriv[j + self.station_offset] -= station_demand*min(x[j + self.station_offset],1)

            # integrate over prices and lost trips
            reward += self.prices[j]*station_demand*min(x[j + self.station_offset],1)
            regret += station_demand*max(1-(min(x[j + self.station_offset],1)),0)

            for start_cell in range(self.n_cells):
                for phase in range(self.n_phases_in[start_cell]):
                    rate = self.in_probabilities[hr][start_cell][j]*self.mu[hr][start_cell][self.cell_idx][phase]*self.phi[hr][start_cell][self.cell_idx][phase]
                    if start_cell == self.cell_idx:
                        deriv[j + self.station_offset] += rate*x[self.x_idx[self.cell_idx]+phase]
                    else:
                        deriv[j + self.station_offset] += rate*self.trajectories[self.x_in_idx[start_cell]+phase][int(t//self.tstep)]
        deriv[-1] = reward
        deriv[-2] = regret

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
