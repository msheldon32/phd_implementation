import numpy as np

# mean value analysis

# 1. take in an array of service times multiplied by routing probabilities

TOL = 0.0001

def mva(routing_array, n_jobs, think_time):
    N = routing_array.shape[0]

    total_service = routing_array.sum(axis=1)
    routing_probs = routing_array / total_service.reshape(N, 1)

    visits = np.zeros(N)
    new_visits = np.ones(N)/N

    while abs((visits-new_visits).max()) > TOL:
        visits = new_visits
        new_visits = visits @ routing_probs
        new_visits = new_visits/sum(new_visits)

    visits = new_visits

    visits = visits * 0.5

    Q = np.zeros(N)
    W = np.zeros(N)
    T = 0

    for job_ct in range(1, n_jobs+1):
        W_q = (1 + Q) / total_service

        weighted_visits = visits * W

        T = job_ct / (weighted_visits.sum() + think_time)

        Q = T * weighted_visits

    return T, Q, W
