from experiment import *
import datetime
import cProfile

def exp_main():
    experiment_config = ExperimentConfig((50, 100))
    experiment        = Experiment(experiment_config)

    now = datetime.datetime.now()

    experiment.output_folder = f"traj_experiment_{now.year}_{now.month}_{now.day}/"

    experiment.run_traj()

if __name__ == "__main__":
    #cProfile.run("exp_main()")
    exp_main()