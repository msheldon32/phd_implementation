import experiment
import datetime

if __name__ == "__main__":
    experiment_config = experiment.ExperimentConfig()
    experiment        = experiment.Experiment(experiment_config)

    now = datetime.datetime.now()

    experiment.output_folder = f"experiment_{now.year}_{now.month}_{now.day}/"

    experiment.run()