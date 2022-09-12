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
import csv
import shutil

import sklearn.cluster

import spatial_decomp_station
import spatial_decomp_strict

import comp
from oslo_data import *
import large_fluid

import pickle

if __name__ == "__main__":
    # load data
    stations = pd.read_csv("oslo2/station_clusters.ipynb")
    n_cells = stations["cell"].max() + 1
    cell_to_station = [[] for i in range(n_cells)]

    for i, row in stations.iterrows():
        cell_to_station[row["cell"]].append[row["index"]]
    
    print(cell_to_station)

    # build StrictTrajCellCox for each station cell


    # run model
