import pandas as pd
import pygambit
from gt_analysis import TputData, CostGenerator

import random

def get_all_levels(possible_levels, n_classes, max_level):
    if n_classes == 1:
        return [[x] for x in possible_levels]
    out_list = [[x] + y for x in possible_levels for y in get_all_levels(possible_levels, n_classes - 1, max_level)]
    return [x for x in out_list]

if __name__ == "__main__":
    solver = pygambit.nash.ExternalEnumPureSolver()

    # random.seed(0)
    
    print("loading data...")
    tput_data = TputData()
    cost_generator = CostGenerator()

    print("data loaded")

    for output_id in range(1,1000):
        print(f"generating game {output_id}...")
        n_classes = random.randint(2, 5)
        print(f"number of players: {n_classes}")
        possible_levels = [_ for _ in range(tput_data.n_jobs+1)]
        all_levels = get_all_levels(possible_levels, n_classes, tput_data.n_jobs)

        game = pygambit.Game.new_table([tput_data.n_jobs+1] * n_classes)

        cost_curves = []

        for player in range(n_classes):
            shape = "linear"#random.choice(["linear", "exp", "convex"])
            cost_curves.append(cost_generator.generate_cost_curve(shape, tput_data.n_jobs+1))

        for level in all_levels:
            total_ct = sum(level)
            if total_ct > tput_data.n_jobs:
                total_tput = tput_data.run_data[output_id][-1]
            else:
                total_tput = tput_data.run_data[output_id][total_ct]
            level_id = [possible_levels.index(x) for x in level]
            for player in range(n_classes):
                player_x = level[player]
                if total_ct == 0:
                    revenue = 0
                else:
                    revenue = player_x * total_tput / total_ct

                game[level_id][player] = round((revenue - cost_curves[player][player_x])*100)



        #for level in all_levels:
        #    print(f"level: {level}")
        #    print(f"payoff for player 0: {game[level][0]}")
        #    print(f"payoff for player 1: {game[level][1]}")

        solutions = solver.solve(game)
        
        accs = set()

        for sol in solutions:
            acc = 0
            n_seen = 0
            player_test_set = set()
            for strat_no, rat in enumerate(sol):
                if rat == 0:
                    continue
                acc += strat_no % (tput_data.n_jobs+1)
                player_test_set.add(strat_no // (tput_data.n_jobs+1))
                n_seen += 1
            if n_seen != n_classes:
                #print(f"did not see all strategies: {n_seen}")
                continue
            
            #assert player_test_set == set(range(n_classes))
            accs.add(acc)

            if acc == 57:
                print("found solution")
                print(sol)
    
        min_acc = min(accs)
        max_acc = max(accs)

        print(f"min acc: {min_acc}")
        print(f"max acc: {max_acc}")

        assert len(accs) > 0

        for acc in range(min_acc, max_acc):
            if acc not in accs:
                assert False

