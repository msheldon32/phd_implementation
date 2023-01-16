import get_oslo_data
import simulator

import pandas as pd
import random

data_folder = "oslo_data_3_big"
default_capacity = 15

if __name__ == "__main__":
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

    start_hour = 5
    end_hour = 21

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

    prices = [1.0 for i in range(n_cells)]
    starting_levels = [10 for i in range(n_cells)]
    

    simulator = simulator.Simulator(station_to_cell, 
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
                for dst_cell, dst_rates in enumerate(src_rates):
                    for phase_idx, rate in enumerate(dst_rates):
                        acc += rate / cum_delay_rate
                        if c_prob <= acc:
                            simulator.process_delay_transition(hour_idx, src_cell, dst_cell, phase_idx)
                            break
    print(f"Finished simulation at t={t}")
    print(f"Total bounces: {simulator.n_bounces}")
    print(f"Total arrivals: {simulator.n_arrivals}")
    print(f"Total profits: {simulator.total_profit}")
