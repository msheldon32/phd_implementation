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


class LargeFluid:
    def __init__(self, n_cells, clusters, durations, in_probabilities, out_demands):
        self.n_cells = n_cells
        self.n_stations = clusters["station"].max()+1

        self.clusters = clusters
        self.durations = durations
        self.out_demands = out_demands
        self.in_probabilities = in_probabilities
    
        self.cell_map = [0 for i in range(self.n_stations)]

        for i, row in clusters.iterrows():
            self.cell_map[int(row["station"])] = int(row["cell"])

    
    def dxdt(self, t, x):
        # first n_cells^2: delay from start_cell to end_cell
        deriv = [0 for i in range((self.n_cells**2)+self.n_stations)]
        
        # input into each delay from each station
        for end_cell in range(self.n_cells):
            for station_idx in range(self.n_stations):
                start_cell = self.cell_map[station_idx]

                delay_d_idx   = (start_cell*self.n_cells) + end_cell
                station_d_idx = (self.n_cells**2) + station_idx

                deriv[delay_d_idx] += self.out_demands[station_idx][end_cell]*min(x[station_d_idx],1)

        # output rate of each delay
        for start_cell in range(self.n_cells):
            for end_cell in range(self.n_cells): 
                delay_d_idx   = (start_cell*self.n_cells) + end_cell
                deriv[delay_d_idx] -= (1/self.durations[start_cell][end_cell])*x[delay_d_idx]
                
        # input into each station from each delay
        for station_idx in range(self.n_stations):
            station_d_idx = (self.n_cells**2) + station_idx
            
            end_cell = self.cell_map[station_idx]

            for start_cell in range(self.n_cells): 
                delay_d_idx = (start_cell*self.n_cells) + end_cell
                rate = self.in_probabilities[start_cell][station_idx]*(1/self.durations[start_cell][end_cell])

                deriv[station_d_idx] += rate*x[delay_d_idx]
            deriv[station_d_idx] -= sum(self.out_demands[station_idx])*min(x[station_d_idx],1)

        return deriv