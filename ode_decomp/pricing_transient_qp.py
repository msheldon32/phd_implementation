import get_oslo_data

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

data_folder = "oslo_data_4"

if __name__ == "__main__":
    # Based on Wollenstein-Betech, this pricing model involves maximizing
    #     sum { lambda_{ij} (p_ij) - c_{ij} lambda_{ij} - c^c {lambda^0_{ij} - lambda_{ij}(u_ij}) }
    #     s.t. sum{lambda_{ij}} = sum{lambda_{ji}}

    # Our model does not feature operational costs per trip, and the lost revenue is simply equal to (1-p)

    # thus we get

    #    max { p_{ij} sum {lambda_{ij} (p_ij)}}
    #     s.t. sum{lambda_{ij}} = sum{lambda_{ji}}

    stations = pd.read_csv(f"{data_folder}/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_stations = int(stations["index"].max()) + 1
    n_cells = int(stations["cell"].max()) + 1

    cell_to_station = [[] for i in range(n_cells)]
    station_to_cell = {}

    start_hour = 5
    end_hour = 21
    hr_range = (end_hour + 1) - start_hour

    output_folder = "qp_prices/"


    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])

    station_demands = [0 for i in range(n_stations**2 * hr_range)]

    for hr in range(start_hour, end_hour+1):
        hr = hr % 24
        hr_idx = hr - start_hour
        in_demands_hr, in_probabilities_hr, out_demands_hr = get_oslo_data.get_demands(hr, cell_to_station, station_to_cell)

        for start_station in range(n_stations):
            start_station_idx = cell_to_station[station_to_cell[start_station]].index(start_station)
            for end_station in range(n_stations):
                start_cell = station_to_cell[start_station]
                end_cell = station_to_cell[end_station]

                end_station_idx = cell_to_station[end_cell].index(end_station)


                station_demands[(hr_idx*n_stations**2)+(start_station*n_stations)+end_station] = out_demands_hr[start_cell][start_station_idx][end_cell] * in_probabilities_hr[end_cell][start_cell][end_station_idx]

        station_demands = np.array(station_demands) + 0.000000001

    # Now, we optimize p(2-p) * lambda_{ij} for each i,j
    # While ensuring that the sum of all lambda_{ij} is equal to the sum of all lambda_{ji}
    m = gp.Model("pricing")
    prices = m.addMVar(shape=hr_range*n_stations**2, name="prices")
    m.setObjective(2*(prices @ station_demands) -  ((prices**2) @ station_demands), GRB.MAXIMIZE)

    for hr in range(start_hour, end_hour+1):
        for station in range(n_stations):
            mask_in = np.zeros(n_stations**2 * hr_range)
            mask_out = np.zeros(n_stations**2 * hr_range)
            for other_station in range(n_stations):
                mask_in[(hr-start_hour)*n_stations**2 + station*n_stations + other_station] = 0.5
                mask_in[((hr-start_hour+hr_range-1)%hr_range)*n_stations**2 + station*n_stations + other_station] = 0.5
                mask_out[(hr-start_hour)*n_stations**2 + other_station*n_stations + station] = 0.5
                mask_out[((hr-start_hour+1)%hr_range)*n_stations**2 + other_station*n_stations + station] = 0.5
                
            m.addConstr(prices @ mask_in == prices @ mask_out)
    # all prices are between 0 and 2
    m.addConstr(prices >= 0)
    m.addConstr(prices <= 2)

    m.optimize()
    print(f"prices for hour {hr}: {prices.X}")

    out_prices = pd.DataFrame(prices.X)
    out_prices["start_station"] = out_prices.index // n_stations
    out_prices["end_station"] = out_prices.index % n_stations

    print(f"max price: {out_prices[0].max()}")
    print(f"min price: {out_prices[0].min()}")
    print(f"average price: {out_prices[0].mean()}")

    out_prices.to_csv(f"{output_folder}/prices.csv")

