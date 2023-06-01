import random

import numpy as np

import scipy.integrate as spi 

import matplotlib.pyplot as plt

class ODESystem:
    def __init__(self, n_eqs, min_rate, max_rate, min_self_rate, max_self_rate):
        self.n_eqs = n_eqs

        self.min_rate = min_rate
        self.max_rate = max_rate


        self.rates = [[random.uniform(min_rate, max_rate) if i != j else random.uniform(min_self_rate, max_self_rate) for j in range(n_eqs)] for i in range(n_eqs)]

        self.is_min = [random.choice([True, False]) for _ in range(n_eqs)]

    def __call__(self, t, y):
        res = [0 for _ in range(self.n_eqs)]

        for i in range(self.n_eqs):
            for j in range(self.n_eqs):
                if i != j:
                    if self.is_min[i]:
                        res[i] += self.rates[i][j] * min(1, y[j])
                    else:
                        res[i] += self.rates[i][j] * y[j]
                elif self.is_min[i]:
                    res[i] -= self.rates[i][j] * min(1, y[j])
                else:
                    res[i] -= self.rates[i][j] * y[j]

        return np.array(res)

if __name__ == "__main__":
    # run 100000 iterations of random ODE systems and ensure all values are non-negative

    TOL = 1e-5

    all_results = np.array([])
    
    n_iterations = 300

    for iter_no in range(n_iterations):
        if iter_no % 100 == 0:
            print("Iteration: {}".format(iter_no))
        n_eqs = random.randint(2, 20)
        ode = ODESystem(n_eqs, 0.1, 10, 0.1, 100)

        y0 = np.array([random.randint(0,50) for _ in range(n_eqs)])

        t = 0
        dt = 10

        x_t = spi.solve_ivp(ode, (t, t + dt), y0, method="RK45", rtol=1e-6, atol=1e-6)

        all_results = np.concatenate((all_results, x_t.y[:, -1]))

        if not np.all(x_t.y[:, -1] >= -TOL):
            print("Negative value found")
            print(f"min: {np.min(x_t.y[:, -1])}")
            assert False


    # plot histogram of results
    plt.hist(all_results, bins=100, range=(0, 1000))
    plt.title(f"Histogram of final values - {n_iterations} iterations")
    plt.show()
