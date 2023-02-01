import get_oslo_data
from simulator import *

import pandas as pd
import numpy as np
import random

data_folder = "oslo_data_3_big"
default_capacity = 15

def get_data():
    stations = pd.read_csv(f"{data_folder}/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = int(stations["cell"].max())+1
    n_stations = len(list(stations["index"]))
    cell_to_station = [[] for i in range(n_cells)]
    #station_to_cell = [0 for i in range(n_stations)]
    station_to_cell = {}
    #capacities = [[] for i in range(n_cells)]
    capacities = [default_capacity for i in range(n_stations)]

    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])
        #capacities[int(row["cell"])].append(int(row["capacity"]))
        capacities[int(row["index"])] = int(row["capacity"])


    mu = []
    phi = []
    in_demands = []
    in_probabilities = []
    out_demands = []

    for hr in range(start_hour, end_hour+1): # need an extra hour of data since solve_ivp sometimes calls the end of the time interval
        hr = hr % 24
        in_demands_hr, in_probabilities_hr, out_demands_hr = get_oslo_data.get_demands(hr, cell_to_station, station_to_cell)

        # phase processes
        mu_hr, phi_hr = get_oslo_data.get_cox_data(hr, len(cell_to_station))

        in_demands.append(in_demands_hr)
        in_probabilities.append(in_probabilities_hr)
        out_demands.append(out_demands_hr)
        
        mu.append(mu_hr)
        phi.append(phi_hr)
    return cell_to_station, station_to_cell, capacities, mu, phi, in_demands, in_probabilities, out_demands, n_cells, n_stations, stations

def run_simulation_epoch(data_list, prices, starting_levels):
    cell_to_station, station_to_cell, capacities, mu, phi, in_demands, in_probabilities, out_demands, n_cells, n_stations, stations = data_list

    simulator = Simulator(station_to_cell, 
                                    cell_to_station, 
                                    n_cells, 
                                    stations, 
                                    mu,
                                    phi,
                                    in_demands,
                                    in_probabilities,
                                    out_demands,
                                    starting_levels,
                                    prices,
                                    capacities)
    t = 0 # not the actual hour, but the index of the hour
    cur_hour = -1
    while not simulator.is_finished(t):
        hour_idx = simulator.get_hour_idx(t)
        if hour_idx != cur_hour:
            print(f"simulating hour: {hour_idx+start_hour}")
            cur_hour = hour_idx
        stn_rates, cum_stn_rate, delay_rates, cum_delay_rate = simulator.get_rates(hour_idx)

        cum_stn_prob = cum_stn_rate / (cum_stn_rate + cum_delay_rate)

        if random.random() <= cum_stn_prob:
            # station event
            c_prob = random.random()
            acc = 0

            for stn_idx, rate in enumerate(stn_rates):
                acc += rate / cum_stn_rate
                if c_prob <= acc:
                    simulator.process_station_transition(hour_idx, stn_idx)
                    break
        else:
            # delay event
            c_prob = random.random()
            acc = 0

            for src_cell, src_rates in enumerate(delay_rates):
                is_finished_src = False
                for dst_cell, dst_rates in enumerate(src_rates):
                    is_finished_dst = False
                    for phase_idx, rate in enumerate(dst_rates):
                        acc += rate / cum_delay_rate
                        if c_prob <= acc:
                            simulator.process_delay_transition(hour_idx, src_cell, dst_cell, phase_idx)
                            is_finished_src = True
                            is_finished_dst = True
                            break
                    if is_finished_dst:
                        break
                if is_finished_src:
                    break
        t = simulator.get_next_time(t, cum_stn_rate+cum_delay_rate)

    rebalancing_costs = simulator.get_rebalancing_costs(1.0)

    print(f"Finished simulation at t={t}")
    print(f"Total bounces: {simulator.n_bounces}")
    print(f"Total arrivals: {simulator.n_arrivals}")
    print(f"Total revenue: {simulator.total_revenue}")
    print(f"Rebalancing cost: {rebalancing_costs}")

    return [simulator.n_bounces, simulator.n_arrivals, simulator.total_revenue, rebalancing_costs]

if __name__ == "__main__":
    start_hour = 5
    end_hour = 21

    n_epochs = 16

    data_list = get_data()
    n_cells, n_stations, stations = data_list[-3:]

    prices = [1.0 for i in range(n_cells)]

    
    prices = [1.2, 0.8999999999999999, 1.3000000000000003, 0.9000000000000002, 0.7000000000000001, 0.6000000000000001, 0.6000000000000001, 1.1, 1.3000000000000003, 1.3000000000000003, 1.1000000000000003, 1.3000000000000003, 0.9, 0.8999999999999999, 1.1, 1.2000000000000002, 1.0, 1.2, 1.0000000000000002, 1.3000000000000003, 1.2, 1.4000000000000006, 1.3000000000000003, 0.6000000000000001, 1.5000000000000007, 0.7, 1.0, 1.4000000000000004, 0.9, 1.1, 1.0, 1.2000000000000002, 1.0000000000000002, 1.2000000000000002, 1.2, 1.0, 0.6000000000000001, 1.1, 1.2000000000000004, 1.0000000000000002, 1.2000000000000002, 1.2000000000000002, 1.1, 1.6000000000000005, 0.7000000000000001, 0.9, 1.1, 1.1000000000000003, 0.9000000000000001, 1.1000000000000005, 1.0, 0.5, 1.2000000000000002, 1.4000000000000004, 1.0, 0.9999999999999999, 1.0, 0.8000000000000003]
    

    starting_levels = [10 for i in range(n_cells)]

    starting_levels = [8.0, 15.0, 12.0, 3.0, 5.0, 5.0, 7.0, 12.0, 6.0, 11.0, 13.0, 1.0, 13, 9.0, 11.0, 15.0, 15.0, 18.0, 11.0, 16.0, 12.0, 12.0, 3.0, 9.0, 6.0, 5.0, 7.0, 10.0, 13.0, 16.0, 7.0, 14.0, 10.0, 12, 11.0, 15, 12, 12.0, 18.0, 5.0, 13.0, 18.0, 14.0, 9.0, 9, 16.0, 15.0, 14.0, 6.0, 11.0, 14.0, 6.0, 5.0, 15.0, 10.0, 14.0, 15.0, 7.0]
    


    bounces = []
    arrivals = []
    revenue = []
    rebalancing_costs = []

    for i in range(n_epochs):
        print(f"Epoch {i+1}/{n_epochs}")
        b, a, r, rc = run_simulation_epoch(data_list, prices, starting_levels)
        bounces.append(b)
        arrivals.append(a)
        revenue.append(r)
        rebalancing_costs.append(rc)

    print(f"Average bounces: {np.mean(bounces)}")
    print(f"Average arrivals: {np.mean(arrivals)}")
    print(f"Average revenue: {np.mean(revenue)}")
    print(f"Average rebalancing costs: {np.mean(rebalancing_costs)}")

    print(f"Bounces: {bounces}")
    print(f"Arrivals: {arrivals}")
    print(f"Revenue: {revenue}")
    print(f"Rebalancing costs: {rebalancing_costs}")


