from util.agents.q_learning import QLearningAgent
from util.agents.random_agent import RandomAgent
import util.cdf
import random
import math

def passenger_reservation():
    return random.randrange(0, 100)

class Model:
    def __init__(self, n_queues, min_rate, max_rate, arrival_rate, agent, holding_cost):
        self.queues = [0 for i in range(n_queues)]
        self.queue_rates = [random.randrange(min_rate, max_rate) for i in range(n_queues)]
        self.arrival_rate = arrival_rate
        self.total_rate = sum(self.queue_rates) + self.arrival_rate
        self.n_queues = n_queues
        self.t = 0
        self.agent = agent
        self.holding_cost = holding_cost
        self.last_price = 0
    
    def step(self):
        total_rate = 0
        action_rates = []

        for i, q in enumerate(self.queues):
            if q == 0:
                action_rates.append(0)
            else:
                action_rates.append(self.queue_rates[i])
                total_rate += self.queue_rates[i]

        action_rates.append(self.arrival_rate)
        total_rate += self.arrival_rate

        action = util.cdf.generate_value(action_rates, random.random())
        t_delta = -math.log(random.random())/total_rate

        self.agent.reinforce(self.last_price-(t_delta*self.holding_cost), tuple(self.queues))
        self.last_price = 0
        
        self.t += t_delta

        if action == self.n_queues:
            # arrival
            self.last_price, queue = self.agent.get_action()
            self.queues[queue] += 1
        elif sum(self.queues) == 0:
            return
        else:
            # departure from queue [action]
            self.queues[action] -= 1

if __name__ == "__main__":
    total_q_return = 0
    total_runs = 10
    total_run_len = 100000
    total_price = 0

    for i in range(total_runs):
        n_queues = 5
        my_agent = QLearningAgent([(a,b) for a in range(100) for b in range(n_queues)])

        model = Model(n_queues, 1, 10, 10, my_agent, 10)
        for i in range(total_run_len):
            model.step()
        total_q_return += my_agent.total_reward
        total_price += my_agent.get_action()[0]
    total_rand_return = 0

    for i in range(total_runs):
        n_queues = 5
        my_agent = RandomAgent([(a,b) for a in range(100) for b in range(n_queues)])

        model = Model(n_queues, 1, 10, 10, my_agent, 10)
        for i in range(total_run_len):
            model.step()
        total_rand_return += my_agent.total_reward
    print("Q return: {}".format(total_q_return/total_runs))
    print("Q price: {}".format(total_price/total_runs))
    print("Random return: {}".format(total_rand_return/total_runs))
