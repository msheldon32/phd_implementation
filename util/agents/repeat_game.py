import random_agent
import q_learning

if __name__=="__main__":
    n_epochs = 1000000

    n_states = 10
    q_agent = q_learning.QLearningAgent([i for i in range(n_states)], learning_rate=0.2)
    q_agent.set_state(0)

    for i in range(n_epochs):
        state = q_agent.state
        if state == (n_states):
            action = q_agent.get_action()
            #print("Agent chooses {} in state {}".format(action, state))
            q_agent.reinforce(-5, action)
        else:
            action = q_agent.get_action()
            #print("Agent chooses {} in state {}".format(action, state))
            if action == state:
                q_agent.reinforce(state, state)
            else:
                #q_agent.reinforce(-20, n_states)
                q_agent.reinforce(-state, n_states)
    print(q_agent.total_reward)
    print("preferred_state: {}".format(q_agent.get_action(n_states)))
    for i in range(n_states):
        stay = q_agent.get_action(i) == i
        print("({}): {}".format(i, stay))
