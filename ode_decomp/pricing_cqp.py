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
    capacity = np.array(stations["capacity"])
    n_stations = int(stations["index"].max()) + 1
    n_cells = int(stations["cell"].max()) + 1

    cell_to_station = [[] for i in range(n_cells)]
    station_to_cell = {}

    start_hour = 5
    end_hour = 21
    hr_range = (end_hour + 1) - start_hour

    output_folder = "cqp_prices/"


    for i, row in stations.iterrows():
        cell_to_station[int(row["cell"])].append(int(row["index"]))
        station_to_cell[int(row["index"])] = int(row["cell"])

    station_demands = [0 for i in range(n_stations**2 * hr_range)]
    total_station_demands = [[0 for i in range(n_stations)] for j in range(n_stations)]
    demand_upto_hour = [[[0 for i in range(n_stations)] for j in range(n_stations)] for k in range(hr_range)]

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
                total_station_demands[start_station][end_station] += out_demands_hr[start_cell][start_station_idx][end_cell] * in_probabilities_hr[end_cell][start_cell][end_station_idx]

        station_demands = np.array(station_demands) + 0.000000001
        demand_upto_hour[hr-start_hour] = np.array(total_station_demands) + 0.000000001

    total_station_demands = np.array(total_station_demands) + 0.000000001


    # Now, we optimize p(2-p) * lambda_{ij} for each i,j
    # While ensuring that the sum of all lambda_{ij} is equal to the sum of all lambda_{ji}
    m = gp.Model("pricing")
    prices = m.addMVar(shape=n_stations, name="prices")
    x0 = m.addMVar(shape=n_stations, name="x0")
    m.setObjective(((2*(prices @ total_station_demands)) - (((prices ** 2) @ total_station_demands))).sum(), GRB.MAXIMIZE)

    m.addConstr((2-prices) @ total_station_demands == ((2-prices) * np.identity(n_stations)) @ total_station_demands @ np.ones(n_stations))

    for hr in range(start_hour, end_hour):
        hr_idx = hr - start_hour
        m.addConstr(x0 + (2-prices) @ demand_upto_hour[hr_idx] - ((2-prices) * np.identity(n_stations)) @ demand_upto_hour[hr_idx] @ np.ones(n_stations) >= np.ones(n_stations))
        m.addConstr(x0 + (2-prices) @ demand_upto_hour[hr_idx] - ((2-prices) * np.identity(n_stations)) @ demand_upto_hour[hr_idx] @ np.ones(n_stations) <= capacity)
    
    # all prices are between 0 and 2
    m.addConstr(prices >= 0)
    m.addConstr(prices <= 2)

    m.optimize()
    print(f"prices for hour {hr}: {prices.X}")

    out_prices = pd.DataFrame(prices.X)
    out_prices["station"] = out_prices.index
    out_prices["x0"] = np.array(x0.X)

    print(f"max price: {out_prices[0].max()}")
    print(f"min price: {out_prices[0].min()}")
    print(f"average price: {out_prices[0].mean()}")

    print(f"max x0: {out_prices['x0'].max()}")
    print(f"min x0: {out_prices['x0'].min()}")
    print(f"average x0: {out_prices['x0'].mean()}")

    out_prices.to_csv(f"{output_folder}/prices.csv")

