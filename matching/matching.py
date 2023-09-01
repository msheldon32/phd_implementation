import numpy as np

import random

class PH:
    def __init__(self, n_phases, generator, starting_prob):
        self.n_phases = n_phases
        self.generator = generator
        self.starting_prob = starting_prob

    def realize(self):
        t = 0

class MatchingGraph:
    def __init__(self, n, arrival_processes):
        self.n = n
        self.adj = np.zeros((n, n), dtype=np.int32)
        self.arrival_processes = arrival_processes
