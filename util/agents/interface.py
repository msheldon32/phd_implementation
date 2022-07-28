import random_agent
import q_learning

if __name__=="__main__":
    actions = []
    while True:
        next_action = input("Enter the next action (-1 to stop): ")
        if next_action == "-1":
            break
        else:
            actions.append(next_action)
    q_agent = q_learning.QLearningAgent(actions)

    starting_state = input("Enter the starting_state: ")
    q_agent.set_state(starting_state)

    while True:
        print("The agent takes the action {} in state {}".format(q_agent.get_action(), q_agent.state))
        next_state = input("Enter the next state (-1 to stop): ")
        if next_state == "-1":
            break
        reinforcement = float(input("Enter the reinforcement: "))
        q_agent.reinforce(reinforcement, next_state)
