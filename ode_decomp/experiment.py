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
import json
import scipy.integrate as spi
import time
import copy

import spatial_decomp_station
import comp
from oslo_data import *
import large_fluid


class ExperimentConfig:
    def __init__(self):
        # parameters
        self.ode_methods             = ["BDF", "RK45"]
        self.stations_per_cell       = [5, 10, 15, 20] 

        # random configuration
        self.n_station_range         = [50, 100]
        self.x_location_range        = [0, 1]
        self.y_location_range        = [0, 1]
        self.station_demand_range    = [0, 0.5]
        self.noise_moments_distance  = [0, 0.2]
    
    def run(self):
        pass

class Experiment:
    def __init__(self):
        pass
    
    def run(self):
        pass