import util.agents.agent as agent
import random

class QLearningAgent(agent.Agent):
    def __init__(self, actions, learning_rate=0.05, epsilon=0.05, discount_rate=0.95):
        super().__init__(actions)
        self.q_function = {}
        self.action = None
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.prev_state = None

    def get_max_q(self, state="default"):
        if state == "default":
            state = self.state

        max_q = float("-inf")
        if state in self.q_function.keys():
            for action in self.actions:
                if action in self.q_function[state].keys():
                    max_q = max(max_q, self.q_function[state][action])
                else:
                    max_q = max(max_q, 0)
        else:
            return 0
        return max_q

    def reinforce(self, reinforcement, new_state):
        if self.state in self.q_function.keys():
            if self.action in self.q_function[self.state]:
                prev_q = self.q_function[self.state][self.action]
                self.q_function[self.state][self.action] = prev_q + self.learning_rate*(self.discount_rate*self.get_max_q(new_state) - prev_q + reinforcement)
            else:
                self.q_function[self.state][self.action] = self.learning_rate*(self.discount_rate*self.get_max_q(new_state) + reinforcement)
        else:
            self.q_function[self.state] = {
                    self.action: self.learning_rate*(self.discount_rate*self.get_max_q(new_state) + reinforcement)
                }
        self.state = new_state
        self.total_reward += reinforcement

    def get_action(self, state="default"):
        if state == "default":
            state = self.state

        if random.random() < self.epsilon:
            self.action = random.choice(self.actions)
        else:
            if state in self.q_function.keys():
                max_q = float("-inf")
                best_action = self.actions[0]
                for action in self.actions:
                    if action in self.q_function[state].keys():
                        if self.q_function[state][action] > max_q:
                            max_q = self.q_function[state][action]
                            best_action = action
                    else:
                        if max_q < 0:
                            max_q = 0
                            best_action = action
                self.action = best_action
            else:
                self.action = random.choice(self.actions)
        return self.action
