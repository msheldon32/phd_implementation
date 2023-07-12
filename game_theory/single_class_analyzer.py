import pandas as pd
import pygambit
from gt_analysis import TputData, CostGenerator

import random
import sys

import csv

MAX_LEVEL = 15
MIN_CLASSES = 2
MAX_CLASSES = 4
STARTING_OUTPUT = 4449
N_OUTPUTS = 5000-STARTING_OUTPUT

def get_all_levels(possible_levels, n_classes, max_level):
    if n_classes == 1:
        return [[x] for x in possible_levels]
    out_list = [[x] + y for x in possible_levels for y in get_all_levels(possible_levels, n_classes - 1, max_level)]
    return [x for x in out_list]

def filter_levels(levels):
    return levels
    # remove levels where there is an x such that x has at least one more than half of the total number of jobs
    #filtered_levels = []

    #for level in levels:
    #    if max(level) <= int((0.5 * sum(level)))+1:
    #        filtered_levels.append(level)

    #return filtered_levels

if __name__ == "__main__":
    output_file = f"{sys.argv[1]}/{sys.argv[2]}.csv"
    cost_curve_output = f"{sys.argv[1]}/cost_curve_{sys.argv[2]}.csv"

    single_prices = int(sys.argv[3]) == 1

    solver = pygambit.nash.ExternalEnumPureSolver()

    # random.seed(0)
    
    print("loading data...")
    tput_data = TputData()
    cost_generator = CostGenerator()

    print("data loaded")
    with open(output_file, "a") as f:
        if single_prices:
            f.write(f"output_id,n_classes,price,min_acc,max_acc,social_optimum_index,soc_stability,soc_anarchy,soc_optimum, min_tput, max_tput, opt_tput\n")
        else:
            f.write(f"output_id,n_classes,x_min,x_max,min_acc,max_acc,social_optimum_index,soc_stability,soc_anarchy,soc_optimum, throughput_ratio, min_tput, max_tput, opt_tput\n")


    for output_id in range(STARTING_OUTPUT, STARTING_OUTPUT+N_OUTPUTS):
        print(f"generating game {output_id}...")
        n_classes = random.randint(MIN_CLASSES, MAX_CLASSES)

        print(f"number of players: {n_classes}")
        if single_prices:
            min_level = 1
            max_level = MAX_LEVEL
        else:
            min_level = random.randint(1, 10)

            max_level = min(MAX_LEVEL, ((n_classes-1)*min_level) + 1)
        span = max_level - min_level + 1
        print(f"strategies are between {min_level} and {max_level}")
        possible_levels = [_ for _ in range(min_level, max_level+1)]
        all_levels = get_all_levels(possible_levels, n_classes, tput_data.n_jobs)
        all_levels = filter_levels(all_levels)

        if single_prices:
            price = random.random()*5.0

        # ensure concavity of run data
        first_differences = [tput_data.run_data[output_id][x+1] - tput_data.run_data[output_id][x] for x in range(len(tput_data.run_data[output_id])-1)]
        second_differences = [first_differences[x+1] - first_differences[x] for x in range(len(first_differences)-1)]

        if any([x > 0 for x in second_differences]):
            print(f"run data: {tput_data.run_data[output_id]}")
            print("not concave")
            continue

        game = pygambit.Game.new_table([(max_level-min_level+1)] * n_classes)

        cost_curves = []

        for player in range(n_classes):
            if single_prices:
                shape = "linear"
                cost_curves.append(cost_generator.generate_cost_curve("linear", tput_data.n_jobs+1, price))
            else:
                shape = random.choice(["linear", "exp", "convex"])
                cost_curves.append(cost_generator.generate_cost_curve(shape, tput_data.n_jobs+1))
 
            with open(cost_curve_output, "a") as f:
                f.write(f"{output_id},{n_classes},{player},{shape},{','.join([str(x) for x in cost_curves[player]])}\n")

        social_optimum = float("-inf")
        optimum_index  = 0
        opt_tput = 0

        for level in all_levels:
            total_ct = sum(level)
            if total_ct > tput_data.n_jobs:
                total_tput = tput_data.run_data[output_id][-1]
            else:
                total_tput = tput_data.run_data[output_id][total_ct]
            level_id = [possible_levels.index(x) for x in level]
            total_reward = 0
            for player in range(n_classes):
                player_x = level[player]
                if total_ct == 0:
                    revenue = 0
                else:
                    revenue = player_x * total_tput / total_ct

                game[level_id][player] = round((revenue - cost_curves[player][player_x])*10000000)
                total_reward += revenue - cost_curves[player][player_x]
            if total_reward >= social_optimum:
                social_optimum = total_reward
                optimum_index = max(optimum_index, sum(level))
                opt_tput = max(total_tput, opt_tput)
            



        #for level in all_levels:
        #    print(f"level: {level}")
        #    print(f"payoff for player 0: {game[level][0]}")
        #    print(f"payoff for player 1: {game[level][1]}")

        solutions = solver.solve(game)
        min_soc = float("inf")
        max_soc = float("-inf")
        min_tput = float("inf")
        max_tput = float("-inf")
        
        accs = set()

        for sol in solutions:
            acc = 0
            n_seen = 0
            
            social_value = 0

            for strat_no, rat in enumerate(sol):
                if rat == 0:
                    continue
                player_val = (strat_no % span) + min_level
                acc += player_val
                n_seen += 1
                social_value -= cost_curves[strat_no // span][player_val]
            if n_seen != n_classes:
                #print(f"did not see all strategies: {n_seen}")
                continue
            if acc >= len(tput_data.run_data[output_id]):
                social_value += tput_data.run_data[output_id][-1]
            else:
                social_value += tput_data.run_data[output_id][acc]
            accs.add(acc)
            min_soc = min(min_soc, social_value)
            max_soc = max(max_soc, social_value)
            if acc >= len(tput_data.run_data[output_id]):
                max_tput = max(max_tput, tput_data.run_data[output_id][-1])
                min_tput = min(min_tput, tput_data.run_data[output_id][-1])
            else:
                max_tput = max(max_tput, tput_data.run_data[output_id][acc])
                min_tput = min(min_tput, tput_data.run_data[output_id][acc])
    
        if len(accs) == 0:
            print(f"no solutions found for game {output_id}")
            print(f"run data: {tput_data.run_data[output_id]}")
            print(f"cost curves: {cost_curves}")

            # write all data to file
            with open(f"violation_{output_id}.txt", "w") as f:
                f.write(f"run data: {tput_data.run_data[output_id]}\n")
                f.write(f"cost curves: {cost_curves}\n")
                f.write(f"game: {game}\n")
                f.write(f"solutions: {solutions}\n")
                f.write(f"accs: {accs}\n")
            solutions = solver.solve(game)

            for i in range(20):
                print(f"retrying game {output_id}...")
                accs = set()

                for sol in solutions:
                    acc = 0
                    n_seen = 0
                    for strat_no, rat in enumerate(sol):
                        if rat == 0:
                            continue
                        #acc += strat_no % (tput_data.n_jobs+1)
                        acc += (strat_no % span) + min_level 
                        n_seen += 1
                    if n_seen != n_classes:
                        #print(f"did not see all strategies: {n_seen}")
                        continue
                    
                    accs.add(acc)

        assert len(accs) > 0   # ASSERTION: a pure nash equilibrium exists

        min_acc = min(accs)
        max_acc = max(accs)

        print(f"min acc: {min_acc}")
        print(f"max acc: {max_acc}")

        print(f"social optimum: {optimum_index}")

        with open(output_file, "a") as f:
            if single_prices:
                f.write(f"{output_id},{n_classes},{price},{min_acc},{max_acc},{optimum_index},{min_soc},{max_soc},{social_optimum}, {min_tput}, {max_tput}, {opt_tput}\n")
            else:
                f.write(f"{output_id},{n_classes},{min_level},{max_level},{min_acc},{max_acc},{optimum_index},{min_soc},{max_soc},{social_optimum}, {min_tput}, {max_tput}, {opt_tput}\n")


        #for acc in range(min_acc, max_acc):
        #    assert acc in accs # ASSERTION: uniqueness of pure nash equilibrium

