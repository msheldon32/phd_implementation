import random
import math
import copy

ELASTICITY = 1.0

class Simulator:
    def __init__(self, stn_to_cell, cell_to_stn, n_cells, stations, mu, phi, in_demands, in_probabilities, out_demands, starting_levels, prices):
        self.n_cells = n_cells
        self.stations = stations
        self.n_stations = len(stn_to_cell)
        self.stn_to_cell = stn_to_cell
        self.cell_to_stn = cell_to_stn
        self.mu = mu
        self.phi = phi
        self.in_demands = in_demands
        self.in_probabilities = in_probabilities
        self.out_demands = out_demands

        self.n_hours = len(self.mu)

        self.delay_lvl = [
            [[0 for phase_idx in range(len(self.mu[0][src_cell][dst_cell]))] for dst_cell in range(self.n_cells)] for src_cell in range(self.n_cells)
                ]
        self.station_lvl = [
                starting_levels[self.stn_to_cell[stn_idx]] for stn_idx in range(self.n_stations)
            ]

        self.n_bounces = 0
        self.n_arrivals = 0
        self.total_profit = 0

        self.prices = prices

        self.cache = {}
        self.set_cache()

        self.adjust_out_demands()

    def set_cache(self):
        self.cache["out_demands"] = copy.deepcopy(self.out_demands)

    def uncache(self):
        self.out_demands = self.cache["out_demands"]

    def adjust_out_demands(self):
        self.uncache()

        for hr in range(self.n_hours):
            for src_cell in range(self.n_cells):
                for stn in self.cell_to_stn[src_cell]:
                    for end_cell in range(self.n_cells):
                        self.out_demands[hr][src_cell][stn][end_cell] += self.out_demands[hr][src_cell][stn][end_cell]*ELASTICITY*(1-self.prices[src_cell])


    def get_rates(self, hour_idx):
        stn_rates = []
        cum_stn_rate = 0
        delay_rates = []
        cum_delay_rate = 0

        for stn in range(n_stations):
            cell_idx = stn_to_cell[stn]
            stn_probs.append(sum(self.out_demands[hour_idx][cell_idx][stn]) if self.station_lvl[stn] > 0 else 0)
            cum_stn_prob += stn_probs[-1]

        for src_cell in range(self.n_cells):
            for dst_cell in range(self.n_cells):
                for phase_idx in range(len(self.mu[0][src_cell][dst_cell])):
                    delay_probs.append(self.mu[hour_idx][src_cell][dst_cell][phase_idx] * self.delay_lvl[src_cell][dst_cell][phase_idx])
                    cum_delay_prob += delay_probs[-1]

        return stn_rates, cum_stn_rate, delay_rates, cum_delay_rate

    def get_next_time(self, t, total_rate):
        return t + math.log(random.random())/total_rate
    
    def get_hour_idx(self, t):
        return math.floor(t)
    
    def is_finished(self, t):
        return t >= self.n_hours

    def process_delay_transition(self, hour_idx, src_cell, dst_cell, phase_idx):
        """
            This function updates the state after a phase transition at a delay.
        """
        is_depature = random.random() <= self.phi[hour_idx][src_cell][dst_cell][phase_idx]

        if is_depature:
            self.delay_lvl[src_cell][dst_cell][phase_idx] -= 1
            c_prob = 0
            x = random.random()
            for stn in self.cell_to_stn[dst_cell]:
                c_prob += self.in_probabilities[hour_idx][dst_cell][src_cell][stn]
                if x <= c_prob:
                    self.station_lvl[stn] += 1
                    break
        else:
            self.delay_lvl[src_cell][dst_cell][phase_idx]   -= 1
            self.delay_lvl[dst_cell][src_cell][phase_idx+1] += 0

    def process_station_transition(self, hour_idx, stn_idx):
        """
           This function updates the state after a departure at each station.
        """
        cell_idx = self.stn_to_cell[stn_idx]
        c_prob = 0
        x = random.random()
        total_out_demand = sum(self.out_demands[hour_idx][cell_idx])

        for dst_cell in range(self.n_cells):
            c_prob += self.out_demands[hour_idx][cell_idx][dst_cell]/total_out_demand
            if x <= c_prob:
                self.station_lvl[stn_idx] -= 1
                self.delay_lvl[cell_idx][dst_cell][0] += 1
                break
