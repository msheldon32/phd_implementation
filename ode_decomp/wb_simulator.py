import random
import math
import copy
import pandas as pd
import csv

ELASTICITY = 1.0

prices_folder = "cqp_prices"
demands_folder = "wb_prices"

single_price = True
no_price = False

class Simulator:
    def __init__(self, stn_to_cell, cell_to_stn, n_cells, stations, mu, phi, starting_ptg, capacities, in_probabilities):
        self.start_hour = 5
        self.n_hours = 17

        self.n_cells = n_cells
        self.stations = stations
        self.n_stations = len(stn_to_cell)
        self.stn_to_cell = stn_to_cell
        self.cell_to_stn = cell_to_stn
        self.mu = mu
        self.phi = phi
        self.capacities = capacities
        self.in_probabilities = in_probabilities
        self.starting_lvl = [0.5 for _ in range(self.n_stations)]


        assert len(self.capacities) == self.n_stations

        self.n_hours = len(self.mu)

        self.delay_lvl = [
            [[0 for phase_idx in range(len(self.mu[0][self.stn_to_cell[src_stn]][self.stn_to_cell[dst_stn]]))]
             for dst_stn in range(self.n_stations)]
            for src_stn in range(self.n_stations)
        ]

        def get_starting_level(stn):
            cell_idx = self.stn_to_cell[stn]
            cell_ptg = starting_ptg[cell_idx]
            int_lvl = math.floor(cell_ptg * self.capacities[stn])
            if random.random() < (cell_ptg * self.capacities[stn] - int_lvl):
                return int_lvl + 1
            return int_lvl

        self.station_lvl = [
               get_starting_level(stn_idx) for stn_idx in range(self.n_stations)
            ]
        

        for i, stn_lvl in enumerate(self.station_lvl):
            self.station_lvl[i] = min(self.station_lvl[i], self.capacities[i])
            self.starting_lvl[i] = self.station_lvl[i]
            #assert stn_lvl <= self.capacities[i]

        self.n_bounces = 0
        self.n_arrivals = 0
        self.total_revenue = 0

        
        self.prices = [[] for _ in range(self.n_hours)]
        self.demands = [[] for _ in range(self.n_hours)]

        for hr_idx in range(self.n_hours):
            self.prices[hr_idx] = [[1 for _ in range(self.n_stations)] for _ in range(self.n_stations)]
            for src_stn in range(self.n_stations):
                for dst_stn in range(self.n_stations):
                    self.prices[hr_idx][src_stn][dst_stn] = 1


            self.demands[hr_idx] = [[0 for _ in range(self.n_stations)] for _ in range(self.n_stations)]
            for src_stn in range(self.n_stations):
                for dst_stn in range(self.n_stations):
                    self.demands[hr_idx][src_stn][dst_stn] = 0
            
            if no_price:
                pass
            elif single_price:
                with open(f"{prices_folder}/prices.csv", newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        src_stn = int(float(row[2]))
                        for dst_stn in range(self.n_stations):
                            self.prices[hr_idx][src_stn][dst_stn] = float(row[1])
                        x0 = float(row[3])
                        if random.random() < (x0 - math.floor(x0)):
                            x0 = math.floor(x0) + 1
                        else:
                            x0 = math.floor(x0)
                        self.station_lvl[src_stn] = min(x0, self.capacities[src_stn])
                        self.starting_lvl[src_stn] = self.station_lvl[src_stn]
            else:
                with open(f"{prices_folder}/prices_{hr_idx+self.start_hour}.csv", newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        src_stn = int(float(row[2]))
                        dst_stn = int(float(row[3]))
                        price = float(row[1])
                        self.prices[hr_idx][src_stn][dst_stn] = price
            with open(f"{demands_folder}/demands_{hr_idx+self.start_hour}.csv", newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    src_stn = int(float(row[2]))
                    dst_stn = int(float(row[3]))
                    demand = float(row[1])
                    self.demands[hr_idx][src_stn][dst_stn] = (2-self.prices[hr_idx][src_stn][dst_stn])*demand

    def get_rates(self, hour_idx):
        stn_rates = []
        cum_stn_rate = 0
        delay_rates = []
        cum_delay_rate = 0

        for stn in range(self.n_stations):
            stn_rates.append(sum(self.demands[hour_idx][stn]) if self.station_lvl[stn] > 0 else 0)
            cum_stn_rate += stn_rates[-1]

        for src_stn in range(self.n_stations):
            src_rates = []
            src_cell = self.stn_to_cell[src_stn]
            for dst_stn in range(self.n_stations):
                dst_rates = []
                dst_cell = self.stn_to_cell[dst_stn]
                for phase_idx in range(len(self.mu[hour_idx][src_cell][dst_cell])):
                    dst_rates.append(self.mu[hour_idx][src_cell][dst_cell][phase_idx] * self.delay_lvl[src_stn][dst_stn][phase_idx])
                    cum_delay_rate += dst_rates[-1]
                src_rates.append(dst_rates)
            delay_rates.append(src_rates)

        return stn_rates, cum_stn_rate, delay_rates, cum_delay_rate

    def get_next_time(self, t, total_rate):
        return t - (math.log(random.random())/total_rate)
    
    def get_hour_idx(self, t):
        return math.floor(t)
    
    def is_finished(self, t):
        return t >= self.n_hours

    def process_delay_transition(self, hour_idx, src_stn, dst_stn, phase_idx):
        """
            This function updates the state after a phase transition at a delay.
        """
        src_cell = self.stn_to_cell[src_stn]
        dst_cell = self.stn_to_cell[dst_stn]
        is_departure = random.random() <= self.phi[hour_idx][src_cell][dst_cell][phase_idx]
        is_departure = is_departure or phase_idx == len(self.mu[hour_idx][src_cell][dst_cell]) - 1

        if is_departure:
            self.delay_lvl[src_stn][dst_stn][phase_idx] -= 1
            if self.station_lvl[dst_stn] >= self.capacities[dst_stn]:
                c_prob = 0
                x = random.random()
                new_dst = random.choice(self.cell_to_stn[dst_cell])

                for stn_idx_in_cell, stn_idx in enumerate(self.cell_to_stn[dst_cell]):
                    c_prob += self.in_probabilities[hour_idx][dst_cell][src_cell][stn_idx_in_cell]
                    if x < c_prob:
                        new_dst = stn_idx
                        break

                self.n_bounces += 1
                self.delay_lvl[dst_stn][new_dst][0] += 1
            else:
                self.n_arrivals += 1
                self.station_lvl[dst_stn] += 1
        else:
            self.delay_lvl[src_stn][dst_stn][phase_idx]   -= 1
            self.delay_lvl[src_stn][dst_stn][phase_idx+1] += 1

    def process_station_transition(self, hour_idx, stn_idx):
        """
           This function updates the state after a departure at each station.
        """
        cell_idx = self.stn_to_cell[stn_idx]
        c_prob = 0
        x = random.random()
        total_out_demand = sum(self.demands[hour_idx][stn_idx]) if self.station_lvl[stn_idx] > 0 else 0


        for dst_stn in range(self.n_stations):
            c_prob += self.demands[hour_idx][stn_idx][dst_stn]/total_out_demand
            if x <= c_prob:
                self.station_lvl[stn_idx] -= 1
                self.delay_lvl[stn_idx][dst_stn][0] += 1
                self.total_revenue += self.prices[hour_idx][stn_idx][dst_stn]   
                break

    def get_rebalancing_costs(self, reb_cost):
        total_reb_cost = 0
        for cell_idx in range(self.n_cells):
            for stn_idx_in_cell, stn_idx in enumerate(self.cell_to_stn[cell_idx]):
                total_reb_cost += abs(self.station_lvl[stn_idx] - self.starting_lvl[stn_idx])*reb_cost
        return total_reb_cost
