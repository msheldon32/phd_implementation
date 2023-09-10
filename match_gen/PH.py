class PHDist:
    def __init__(self, starting_probs, transition_matrix):
        self.starting_probs = starting_probs
        self.transition_matrix = transition_matrix

    def get_total_rate(self, state):
        return sum(self.transition_matrix[state])
    
    def transition(self, state, prob):
        c_prob = 0

        for i in range(len(self.transition_matrix[state])):
            c_prob += self.transition_matrix[state][i]

            if prob <= c_prob:
                return i


