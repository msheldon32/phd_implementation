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

from spatial_decomp import *


def twocell_test():
    cell_durations = [
        [random.random(), random.random()],
        [random.random(), random.random()]]

    in_probabilities = [
        [[random.random()],
         [random.random()]] for i in range(2)]
    
    for i in range(2):
        for j in range(2):
            in_probabilities[i][j].append(1-in_probabilities[i][j][0])
    
    out_demands = [
        [[random.random()*10, random.random()*10],
         [random.random()*10, random.random()*10]] for i in range(2)]
    
    get_dxdt0 = get_traj_function(0, cell_durations, in_probabilities[0], out_demands[0])
    get_dxdt1 = get_traj_function(1, cell_durations, in_probabilities[1], out_demands[1])
    
    traj0 = lambda t: [0, 0] # trajectories of delays into 0
    traj1 = lambda t: [0, 0] # trajectories of delays into 1
    
    dxdt0 = get_dxdt0(traj0)
    dxdt1 = get_dxdt1(traj1)
    
    # -------------------------------------------
    # no movement if everything is zero
    # -------------------------------------------
    x0_0 = [0,0,0,0]
    x1_0 = [0,0,0,0]
    assert all([x == 0 for x in dxdt0(0, x0_0)])
    assert all([x == 0 for x in dxdt1(0, x1_0)])
    
    # -------------------------------------------
    # only movement out of delays
    # -------------------------------------------
    x0 = [random.random()*10,random.random()*10,0,0]
    x1 = [random.random()*10,random.random()*10,0,0]
    
    only_delay_0 = dxdt0(0, x0)
    only_delay_1 = dxdt1(0, x1)
    
    # rate out of each delay: (1/duration)* x
    tol = 0.000001
    target = -(x0[0]/cell_durations[0][0])
    assert target-tol < only_delay_0[0] < target+tol
    target = -(x0[1]/cell_durations[0][1])
    assert target-tol < only_delay_0[1] < target+tol
    
    target = -(x1[0]/cell_durations[1][0])
    assert target-tol < only_delay_1[0] < target+tol
    target = -(x1[1]/cell_durations[1][1])
    assert target-tol < only_delay_1[1] < target+tol
    
    # rate into each station: ((1/duration)*delay)*prob
    target = in_probabilities[0][0][0]*(x0[0]/cell_durations[0][0])
    assert target-tol < only_delay_0[2] < target+tol
    target = in_probabilities[0][0][1]*(x0[0]/cell_durations[0][0])
    assert target-tol < only_delay_0[3] < target+tol
    
    target = in_probabilities[1][1][0]*(x1[1]/cell_durations[1][1])
    assert target-tol < only_delay_1[2] < target+tol
    target = in_probabilities[1][1][1]*(x1[1]/cell_durations[1][1])
    assert target-tol < only_delay_1[3] < target+tol
    
    # -------------------------------------------
    # with trajectories
    # -------------------------------------------
    traj2_0 = lambda t: [t,t]
    traj2_1 = lambda t: [math.sqrt(t),math.sqrt(t)]
    dxdt0 = get_dxdt0(traj2_0)
    dxdt1 = get_dxdt1(traj2_1)
    
    x0 = [random.random()*10,random.random()*10,0,0]
    x1 = [random.random()*10,random.random()*10,0,0]
    
    with_traj0 = dxdt0(4, x0)
    with_traj1 = dxdt1(4, x1)
    
    # rate out of each delay: (1/duration)* x
    tol = 0.000001
    target = -(x0[0]/cell_durations[0][0])
    assert target-tol < with_traj0[0] < target+tol
    target = -(x0[1]/cell_durations[0][1])
    assert target-tol < with_traj0[1] < target+tol
    
    target = -(x1[0]/cell_durations[1][0])
    assert target-tol < with_traj1[0] < target+tol
    target = -(x1[1]/cell_durations[1][1])
    assert target-tol < with_traj1[1] < target+tol
    
    # rate into each station: ((1/duration)*delay)*prob
    target = (in_probabilities[0][0][0]*(x0[0]/cell_durations[0][0])) + (in_probabilities[0][1][0]*(4/cell_durations[1][0]))
    assert target-tol < with_traj0[2] < target+tol
    target = in_probabilities[0][0][1]*(x0[0]/cell_durations[0][0]) + (in_probabilities[0][1][1]*(4/cell_durations[1][0]))
    assert target-tol < with_traj0[3] < target+tol
    
    target = in_probabilities[1][1][0]*(x1[1]/cell_durations[1][1]) + (in_probabilities[1][0][0]*(2/cell_durations[0][1]))
    assert target-tol < with_traj1[2] < target+tol
    target = in_probabilities[1][1][1]*(x1[1]/cell_durations[1][1]) + (in_probabilities[1][0][1]*(2/cell_durations[0][1]))
    assert target-tol < with_traj1[3] < target+tol

if __name__ == "__main__":
    twocell_test()