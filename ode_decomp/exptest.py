import experiment

my_stations = [[0.5,0.5],[0.7,0.7],[0.3,0.2]]
my_durations = [[0,1,1],[1,0,1],[1,1,0]]
my_demands = [[0,1,2],[3,0,4],[5,6,0]]
emodel = experiment.ExperimentModel(my_stations, my_durations, my_demands, [5,5,5], 1)
emodel.generate_cells(2)
print(my_stations)
print(my_demands)
print(my_durations)
print(emodel.cell_to_station)
print("---------------------------")
print(emodel.get_durations_strict())
print(emodel.get_in_demands_strict())
print(emodel.get_in_probs_strict())
print(emodel.get_out_demands_strict())
