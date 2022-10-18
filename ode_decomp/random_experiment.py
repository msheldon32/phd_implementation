from experiment import *
import datetime
import cProfile
import sys

def exp_main(seed, control):
    #experiment_config = ExperimentConfig(seed, (50,100), 4)
    experiment_config = ExperimentConfig(seed, (125,150), 4)
    #experiment_config = ExperimentConfig(seed, (20,30), 4)
    experiment        = Experiment(experiment_config)

    now = datetime.datetime.now()

    experiment.output_folder = f"experiment_{now.year}_{now.month}_{now.day}_{seed}/"

    if control:
        experiment.run_control()
    else:
        experiment.run_validation()

if __name__ == "__main__":
    #cProfile.run("exp_main()")
    exp_main(int(sys.argv[1]), int(sys.argv[2]) == 1)
