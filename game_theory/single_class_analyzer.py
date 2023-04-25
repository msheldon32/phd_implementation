import pandas as pd
import pygambit
from gt_analysis import TputData, CostGenerator

import random

def get_all_levels(possible_levels, n_classes, max_level):
    if n_classes == 1:
        return [[x] for x in possible_levels]
    out_list = [[x] + y for x in possible_levels for y in get_all_levels(possible_levels, n_classes - 1, max_level)]
    return [x for x in out_list if sum(x) <= max_level]

if __name__ == "__main__":
    solver = pygambit.nash.ExternalEnumPureSolver()

    # random.seed(0)
    
    print("loading data...")
    tput_data = TputData()
    cost_generator = CostGenerator()

    print("data loaded")

    for output_id in range(1,17):
        print(f"generating game {output_id}...")
        n_classes = random.randint(2, 3)
        print(f"number of players: {n_classes}")
        possible_levels = [_ for _ in range(tput_data.n_jobs)]
        all_levels = get_all_levels(possible_levels, n_classes, tput_data.n_jobs)

        game = pygambit.Game.new_table([len(possible_levels)] * n_classes)

        cost_curves = []

        for player in range(n_classes):
            shape = "convex"#random.choice(["linear", "exp", "convex"])
            cost_curves.append(cost_generator.generate_cost_curve(shape, tput_data.n_jobs))

        for level in all_levels:
            total_ct = sum(level)
            level_id = [possible_levels.index(x) for x in level]
            for player in range(n_classes):
                player_x = level[player]
                if total_ct == 0:
                    revenue = 0
                else:
                    revenue = player_x * tput_data.run_data[output_id][total_ct] / total_ct

                game[level_id][player] = round((revenue - cost_curves[player][player_x])*100)

        #for level in all_levels:
        #    print(f"level: {level}")
        #    print(f"payoff for player 0: {game[level][0]}")
        #    print(f"payoff for player 1: {game[level][1]}")

        solutions = solver.solve(game)

        for sol in solutions:
            print("-----------------------------------------------------------------------------------------")
            acc = 0
            for level, level_soln in enumerate(sol):
                if level_soln > 0:
                    acc += level
                    #print(f"rational level: {level}")
            print(f"acc: {acc}")
