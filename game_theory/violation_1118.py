
if __name__ == "__main__":
    run_data = [0, 7.8577, 13.817, 18.345, 21.77, 24.335, 26.224, 27.59, 28.554, 29.218, 29.663, 29.955, 30.141, 30.257, 30.329, 30.372, 30.397, 30.412, 30.42, 30.425, 30.428]
    costs = [[0, 0.09102894291474366, 0.34372850385149023, 0.6140853324606055, 1.1028988036787393, 1.612564382897271, 2.337691421865146, 3.174358346209721, 4.022372838412239, 4.997444220777019, 6.162485686663713, 7.420298323172634, 8.699867724340306, 10.13980435075319, 11.605065626566946, 13.194236355085547, 14.820482075770494, 16.55900179165391, 18.326263785934728, 20.241427569392197, 22.300754591630305], [0.0, 0.3709438201723897, 0.7418876403447794, 1.1128314605171692, 1.4837752806895588, 1.8547191008619484, 2.2256629210343384, 2.596606741206728, 2.9675505613791175, 3.338494381551507, 3.7094382017238967, 4.080382021896287, 4.451325842068677, 4.822269662241066, 5.193213482413456, 5.564157302585845, 5.935101122758235, 6.306044942930625, 6.676988763103014, 7.047932583275404, 7.418876403447793]]

    level_1 = 0
    level_2 = 0

    while True:
        print(f"current levels: ({level_1}, {level_2})")
        
        if level_1 + level_2 == 0:
            reward_1 = 0
            reward_2 = 0
        elif level_1 + level_2 > 20:
            reward_1 = (level_1 / (level_1 + level_2)) * run_data[-1] - costs[0][level_1]
            reward_2 = (level_2 / (level_1 + level_2)) * run_data[-1] - costs[1][level_2]
        else:
            reward_1 = (level_1 / (level_1 + level_2)) * run_data[level_1 + level_2] - costs[0][level_1]
            reward_2 = (level_2 / (level_1 + level_2)) * run_data[level_1 + level_2] - costs[1][level_2]

        print(f"rewards: ({reward_1}, {reward_2})")
        print(f"costs: ({costs[0][level_1]}, {costs[1][level_2]})")
        
        max_reward_1 = reward_1
        max_reward_2 = reward_2

        argmax_1 = level_1
        argmax_2 = level_2

        for level in range(0, 20):
            if level + level_2 > 20:
                new_reward_1 = (level / (level + level_2)) * run_data[-1] - costs[0][level]
            elif level + level_2 == 0:
                new_reward_1 = 0
            else:
                new_reward_1 = (level / (level + level_2)) * run_data[level + level_2] - costs[0][level]
            
            if level + level_1 > 20:
                new_reward_2 = (level / (level + level_1)) * run_data[-1] - costs[1][level]
            elif level + level_1 == 0:
                new_reward_2 = 0
            else:
                new_reward_2 = (level / (level + level_1)) * run_data[level + level_1] - costs[1][level]
            
            if new_reward_1 > max_reward_1:
                max_reward_1 = new_reward_1
                argmax_1 = level

            if new_reward_2 > max_reward_2:
                max_reward_2 = new_reward_2
                argmax_2 = level

        if argmax_1 == level_1 and argmax_2 == level_2:
            print(f"final levels: ({level_1}, {level_2})")
            break
        
        level_1 = argmax_1
        level_2 = argmax_2

