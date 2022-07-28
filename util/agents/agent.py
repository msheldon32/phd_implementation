import random

class Agent:
    def __init__(self, actions):
        self.total_reward = 0
        self.actions = actions
        self.state = None

    def reinforce(self, reward, new_state):
        self.total_reward += reward
        self.state = new_state

    def set_state(self, new_state):
        self.state = new_state

    def get_action(self, state="default"):
        if state == "default":
            state = self.state
        return random.choice(self.actions)
