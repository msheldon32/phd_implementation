import experiment

import pandas as pd
import numpy as np

def get_durations():
    durations_df = pd.read_csv("oslo3/station_distances.csv")
    n_stations = durations_df["start"].max() + 1
    durations = [[0 for i in range(n_stations)] for j in range(n_stations)]
    for i, row in durations_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        durations[start][end] = row["duration"]/3600
    return durations

def get_demands():
    demands_df = pd.read_csv("oslo3/station_demands.csv")
    n_stations = demands_df["start"].max() + 1
    demands = [[0 for i in range(n_stations)] for j in range(n_stations)]
    for i, row in demands_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        demands[start][end] = row["hourly_demand"]
    return demands


def get_cell_data():
    station_df = pd.read_csv("oslo3/stations.csv").rename({"Unnamed: 0": "index"}, axis=1)
    n_cells = station_df["cell"].max() + 1
    n_stations = station_df["index"].max() + 1
    cell_to_station = [set() for i in range(n_cells)]
    station_to_cell = [0 for i in range(n_stations)]

    for i, row in station_df.iterrows():
        cell_to_station[row["cell"]].add(row["index"])
        station_to_cell[row["index"]] = row["cell"]

    return [n_cells, cell_to_station, station_to_cell]

if __name__ == "__main__":
    n_stations = 291

    exp = experiment.Experiment(experiment.ExperimentConfig(1996, [250, 300], 4))

    stations  = [i for i in range(n_stations)]
    durations = get_durations()
    demands   = get_demands()

    cell_data = get_cell_data()

    model = experiment.ExperimentModel(stations, durations, demands, [6 for i in range(n_stations)], 1)
    model.n_cells = cell_data[0]
    model.cell_to_station = cell_data[1]
    model.station_to_cell = cell_data[2]

    #print([[x for x in y if  x <= 0.1] for y in model.get_durations_strict()])

    res = exp.run_iteration_strict(model, "BDF", [0.1])
    np.savetxt("res.csv",res[0][1], delimiter=",")