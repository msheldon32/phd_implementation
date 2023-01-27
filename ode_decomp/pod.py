import pandas as pd
import numpy as np

import math
import random

class Model:
    def __init__(self, demand, mu, phi, prob, cap):
        # demand: [hour][start_station][end_station]
        self.demand = demand
        # mu: [hour][start_station][end_station][phase]
        self.mu = mu
        # phi: [hour][start_station][end_station][phase]
        self.phi = phi

        self.n_hours = len(self.demand)
        self.n_stations = len(self.demand[0])
        self.cap = cap

class Simulator:
    def __init__(self, model, starting_levels):
        self.model = model
        self.station_levels = starting_levels
        self.delay_levels = [[[0 for _ in range(self.model.mu[src_stn][dst_stn])]
                                for dst_stn in range(self.model.n_stations)] for src_stn in range(self.model.n_stations)]

    def get_transition_rates(self, t):
        hr_idx = math.floor(t)
        station_rates = [sum(self.model.demand[hr_idx][i])*min(self.station_levels[i],1) for i in range(self.model.n_stations)]
        
        total_station_rates = sum(station_rates)

        delay_rates = []
        for src_station in range(self.model.n_stations):
            delay_rates.append([])
            for dst_station in range(self.model.n_stations):
                for phase in range(len(self.model.phi[hr_idx][src_station][dst_station])):
                    delay_rates[-1].append(self.model.phi[hr_idx][src_station][dst_station][phase]*self.model.mu[hr_idx][src_station][dst_station][phase]*self.delay_levels[src_station][dst_station][phase])

        total_delay_rates = sum([sum(x) for x in delay_rates])

        return station_rates, delay_rates, total_station_rates, total_delay_rates
    
    def apply_transition_to_station(self, t, src_station):
        hr_idx = math.floor(t)
        self.station_levels[src_station] -= 1

        total_stn_rate = sum(self.model.demand[hr_idx][src_station])

        X = random.random()

        for dst_station in range(self.model.n_stations):
            X -= self.model.demand[hr_idx][src_station][dst_station]/total_stn_rate

            if X <= 0:
                self.delay_levels[src_station][dst_station][0] += 1
                return
    
    def apply_transition_to_delay(self, t, src_station, dst_station, phase):
        self.delay_levels[src_station][dst_station][phase] -= 1

        if random.random() <= self.model.phi[hr_idx][src_station][dst_station][phase]:
            self.station_levels[dst_station] += 1
        else:
            self.delay_levels[src_station][dst_station][phase+1] += 1
