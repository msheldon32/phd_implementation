import numpy as np
import random

import optuna
import torch

import mva


N_STATIONS = 25
MIN_RATE = 1
MAX_RATE = 100
N_JOBS = 250


class Critic(torch.nn.Module):
    def __init__(self, n_hidden_layers, layer_size, input_size, dropout):
        super(Critic, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.layer_size = layer_size

        layers = [torch.nn.Linear(input_size, layer_size), torch.nn.ReLU(), torch.nn.Dropout(dropout)]

        for i in range(n_hidden_layers):
            layers.append(torch.nn.Linear(layer_size, layer_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))

        output_layer = torch.nn.Linear(layer_size, 1)

        self.stack = torch.nn.Sequential(*(layers + [output_layer]))

    def forward(self, x):
        return self.stack(x)

def generate_demand_matrix():
    return np.random.randint(MIN_RATE, MAX_RATE, (N_STATIONS, N_STATIONS))

def generate_demand_matrix_oos():
    return np.random.randint(MIN_RATE_OOS, MAX_RATE_OOS, (N_STATIONS, N_STATIONS))

def evaluate_demand(demand_matrix):
    return mva.mva(demand_matrix, N_JOBS)[0]

def train(critic, n_samples, lr):
    critic.train()

    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    for _ in range(n_samples):
        demand_matrix = generate_demand_matrix()
        demand = evaluate_demand(demand_matrix)

        in_demand_tensor = torch.tensor(demand_matrix, dtype=torch.float32).flatten()
        demand_tensor = torch.tensor(demand, dtype=torch.float32).flatten()



        optimizer.zero_grad()

        output = critic(in_demand_tensor)

        loss = torch.nn.MSELoss()(output, demand_tensor)

        loss.backward()

        optimizer.step()
    print(f"Training loss: {loss.item()}")
    return critic, loss.item()

def get_val_loss(critic, n_samples):
    critic.eval()
    
    val_loss = 0

    for _ in range(n_samples):
        demand_matrix = generate_demand_matrix()
        demand = evaluate_demand(demand_matrix)

        in_demand_tensor = torch.tensor(demand_matrix, dtype=torch.float32).flatten()
        demand_tensor = torch.tensor(demand, dtype=torch.float32).flatten()



        output = critic(in_demand_tensor)

        loss = torch.nn.MSELoss()(output, demand_tensor)

        val_loss += loss.item()

        if random.random() < 0.01:
            print(f"Machine says: {output.item()}, actual: {demand}, naive: {np.sum(demand_matrix)}, loss: {loss.item()}")

    return val_loss / n_samples

def objective(trial):
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 7)
    layer_size = trial.suggest_int("layer_size", 10, 100)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    #dropout = trial.suggest_float("dropout", 0, 1)
    dropout = 0

    critic = Critic(n_hidden_layers, layer_size, N_STATIONS**2, dropout)

    train(critic, 30000, learning_rate)

    return get_val_loss(critic, 1000)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", storage=f"sqlite:///critic_train_{N_STATIONS}.db", study_name=f"critic_train_{N_STATIONS}", load_if_exists=True)
    study.optimize(objective, n_trials=500)

