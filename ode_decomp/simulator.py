class Simulator:
    def __init__(self, n_cells, stations, mu, phi, in_demands, in_probabilities, out_demands):
        self.n_cells = n_cells
        self.stations = stations
        self.mu = mu
        self.phi = phi
        self.in_demands = in_demands
        self.in_probabilities = in_probabilities
        self.out_demands = out_demands

        self.delay_lvl = [
            [[0 for phase_idx in range(len(self.mu[0][src_cell][dst_cell]))] for dst_cell in range(self.n_cells)] for src_cell in range(self.n_cells)
                ]
        self.station_lvl = [
            0 for stn_idx in range(len(self.stations))
                ]

    def get_rates_and_probabilities(self):
        pass

    def process_delay_transition(self, src_cell, dst_cell, phase_idx):
        pass

    def process_station_transition(self, stn_idx):
        pass
