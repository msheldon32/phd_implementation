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

class FluidModel:
    def __init__(self, mu, phi, trj_mu, trj_phi, trajectories, station_demands, std_prob, dts_prob, tts_prob):
        self.mu = mu
        self.phi = phi

        self.trj_mu = trj_mu
        self.trj_phi = trj_phi

        self.station_demands = station_demands

        self.n_stations = len(station_demands)
        self.n_delays   = len(mu)

        self.std_connections = {}
        self.dts_connections = {}
        self.tts_connections = {}

        self.trajectories = trajectories

        # determine connections
        for stn_id, row in enumerate(std_prob):
            self.std_connections[stn_id] = []
            for del_id, p in enumerate(row):
                if p != 0:
                    self.std_connections[stn_id].append(del_id)

        for del_id, row in enumerate(dts_prob):
            self.dts_connections[del_id] = []
            for stn_id, p in enumerate(row):
                if p != 0:
                    self.dts_connections[del_id].append(stn_id)
        
        for trj_id, row in enumerate(tts_prob):
            self.tts_connections[trj_id] = [] 
            for stn_id, p in enumerate(row):
                if p != 0:
                    self.tts_connections[trj_id].append(stn_id)
        
        # save the offset point for each delay
        self.x_del_idx = []
        cur_offset = 0
        
        for del_id, m in enumerate(self.mu):
            self.x_del_idx.append(cur_offset)
            cur_offset += len(m)
        
        # overall offset point: start of stations
        self.x_stn_idx_start = cur_offset

        # save the offset point for each trajectory
        self.x_trj_idx = []
        cur_offset = 0

        for trj_id, m in enumerate(self.trj_mu):
            self.x_trj_idx.append(cur_offset)
            cur_offset += len(m)
            

    def get_x_idx(self, point_type, idx, phase=0):
        if point_type == "station":
            return self.x_stn_idx_start + idx
        
        return self.x_del_idx[idx] + phase


    def set_trajectories(self, trajectories):
        self.trajectories = trajectories



    def dxdt(self, t, x):
        deriv = [0 for i in range(self.x_stn_idx_start + self.n_stations)]

        # input from stations into delays
        for stn_id, stn_mu in enumerate(self.station_demands):
            stn_x_idx = self.x_stn_idx_start + stn_id
            out_flow = stn_mu*min(1, x[stn_x_idx])
            deriv[stn_x_idx] -= out_flow

            # direct outflow accordingly to delays
            for del_id in self.std_connections[stn_id]:
                del_x_idx = self.x_del_idx[del_id]
                deriv[del_x_idx] += out_flow*std_prob[stn_id][del_id]
        
        # update flow within/from delays
        for del_id in range(self.n_delays):
            n_phases = len(self.mu[del_id])

            del_x_idx = self.x_del_idx[del_id]

            total_outflow = 0

            for phase in range(n_phases):
                phase_flow = x[del_x_idx+phase] * self.mu[del_id][phase]
                deriv[del_x_idx+phase] -= phase_flow

                total_outflow += phase_flow * self.phi[del_id][phase]

            for stn_id in self.std_connections[del_id]:
                stn_x_idx = self.x_stn_idx_start + stn_id
                deriv[stn_x_idx] += total_outflow * self.dts_prob[del_id][stn_id]
        
        # update flow from trajectories
        for trj_id, trj_fn in enumerate(self.trajectories):
            trj_x_idx = self.x_trj_idx[trj_id]

            for stn_id in self.tts_connections[trj_id]:
                stn_x_idx = self.x_stn_idx_start + stn_id

                for phase, (mu, phi) in enumerate(zip(self.trj_mu[trj_id], self.trj_phi[trj_id])):
                    deriv[stn_x_idx] += self.trajectories[trj_x_idx+phase]*mu*phi
        
        return deriv