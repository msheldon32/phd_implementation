import random
import math
import copy

ELASTICITY = 1.0

class Simulator:
    def __init__(self, stn_to_cell, cell_to_stn, n_cells, stations, mu, phi, in_demands, in_probabilities, out_demands, starting_levels, prices, capacities):
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
        self.capacities = capacities

        assert len(self.capacities) == self.n_stations

        self.n_hours = len(self.mu)

        self.delay_lvl = [
            [[0 for phase_idx in range(len(self.mu[0][src_cell][dst_cell]))] for dst_cell in range(self.n_cells)] for src_cell in range(self.n_cells)
                ]
        self.station_lvl = [
                starting_levels[self.stn_to_cell[stn_idx]] for stn_idx in range(self.n_stations)
            ]

        for i, stn_lvl in enumerate(self.station_lvl):
            self.station_lvl[i] = min(self.station_lvl[i], self.capacities[i])
            #assert stn_lvl <= self.capacities[i]

        self.starting_levels = starting_levels

        self.n_bounces = 0
        self.n_arrivals = 0
        self.total_revenue = 0

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
                for stn, real_stn in enumerate(self.cell_to_stn[src_cell]):
                    for end_cell in range(self.n_cells):
                        self.out_demands[hr][src_cell][stn][end_cell] += self.out_demands[hr][src_cell][stn][end_cell]*ELASTICITY*(1-self.prices[src_cell])


    def get_rates(self, hour_idx):
        stn_rates = []
        cum_stn_rate = 0
        delay_rates = []
        cum_delay_rate = 0

        for stn in range(self.n_stations):
            cell_idx = self.stn_to_cell[stn]
            stn_idx_in_cell = self.cell_to_stn[cell_idx].index(stn)
            stn_rates.append(sum(self.out_demands[hour_idx][cell_idx][stn_idx_in_cell]) if self.station_lvl[stn] > 0 else 0)
            cum_stn_rate += stn_rates[-1]

        for src_cell in range(self.n_cells):
            src_rates = []
            for dst_cell in range(self.n_cells):
                dst_rates = []
                for phase_idx in range(len(self.mu[0][src_cell][dst_cell])):
                    dst_rates.append(self.mu[hour_idx][src_cell][dst_cell][phase_idx] * self.delay_lvl[src_cell][dst_cell][phase_idx])
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

    def process_delay_transition(self, hour_idx, src_cell, dst_cell, phase_idx):
        """
            This function updates the state after a phase transition at a delay.
        """
        is_departure = random.random() <= self.phi[hour_idx][src_cell][dst_cell][phase_idx]

        is_departure = is_departure or phase_idx == len(self.mu[hour_idx][src_cell][dst_cell]) - 1

        if is_departure:
            self.delay_lvl[src_cell][dst_cell][phase_idx] -= 1
            c_prob = 0
            x = random.random()
            for stn_idx_in_cell, stn_idx in enumerate(self.cell_to_stn[dst_cell]):
                c_prob += self.in_probabilities[hour_idx][dst_cell][src_cell][stn_idx_in_cell]
                assert stn_idx < self.n_stations
                if x <= c_prob:
                    if self.station_lvl[stn_idx] >= self.capacities[stn_idx]:
                        #print(f"Station is full: {stn_idx}")
                        self.n_bounces += 1
                        #if random.random() < 0.5:
                        #    self.delay_lvl[dst_cell][src_cell][0] += 1
                        #else:
                        self.delay_lvl[dst_cell][dst_cell][0] += 1
                    else:
                        self.n_arrivals += 1
                        self.station_lvl[stn_idx] += 1
                    break
        else:
            self.delay_lvl[src_cell][dst_cell][phase_idx]   -= 1
            self.delay_lvl[src_cell][dst_cell][phase_idx+1] += 1

    def process_station_transition(self, hour_idx, stn_idx):
        """
           This function updates the state after a departure at each station.
        """
        cell_idx = self.stn_to_cell[stn_idx]
        c_prob = 0
        x = random.random()
        stn_idx_in_cell = self.cell_to_stn[cell_idx].index(stn_idx)
        total_out_demand = sum(self.out_demands[hour_idx][cell_idx][stn_idx_in_cell])

        self.total_revenue += self.prices[cell_idx]

        for dst_cell in range(self.n_cells):
            c_prob += self.out_demands[hour_idx][cell_idx][stn_idx_in_cell][dst_cell]/total_out_demand
            if x <= c_prob:
                self.station_lvl[stn_idx] -= 1
                self.delay_lvl[cell_idx][dst_cell][0] += 1
                break

    def get_rebalancing_costs(self, reb_cost):
        total_reb_cost = 0
        for cell_idx in range(self.n_cells):
            for stn_idx_in_cell, stn_idx in enumerate(self.cell_to_stn[cell_idx]):
                total_reb_cost += abs(self.station_lvl[stn_idx] - self.starting_levels[cell_idx]) * reb_cost
        return total_reb_cost
