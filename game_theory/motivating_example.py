import pygambit

from gt_analysis import CostGenerator, TputData

import random
import math

M = 1000
strat_size = 20
n_players = 3

PRICE_EXPONENT = 8

def get_continuous_equilibrium_degen(costs, current_strategies):
    n_players = len(costs)

    min_strat = 1
    max_strat = strat_size 

    for i in range(100000):
        Q = sum(current_strategies)

        next_strategies = [0] * n_players

        for player in range(n_players):
            q = Q - current_strategies[player]
            
            next_strategies[player] = math.sqrt(q/costs[player]) - q
            next_strategies[player] = max(min_strat, min(max_strat, next_strategies[player]))

        if all([x == y for x, y in zip(current_strategies, next_strategies)]):
            break

        current_strategies = next_strategies

    return sum(current_strategies)


if __name__ == "__main__":
    mixed_equilibria = []
    continuous_equilibria = []

    tput_data = TputData()

    for trial in range(100000):
        if trial % 100 == 0:
            print(f"Trial {trial}")
        tput_q = [0, 1]
        tput = tput_data.run_data[trial]

        cost_generator = CostGenerator()

        price_1 = round(random.random()**PRICE_EXPONENT, 2)
        price_2 = round(random.random()**PRICE_EXPONENT, 2)
        price_3 = round(random.random()**PRICE_EXPONENT, 2)
        price_4 = round(random.random()**PRICE_EXPONENT, 2)

        def fix_price(price):
            while price == 0:
                price = round(random.random()**PRICE_EXPONENT, 2)
            return price
        price_1 = fix_price(price_1)
        price_2 = fix_price(price_2)
        price_3 = fix_price(price_3)
        price_4 = fix_price(price_4)
        
        cost_1 = cost_generator.generate_cost_curve("linear", strat_size, price=price_1)
        cost_2 = cost_generator.generate_cost_curve("linear", strat_size, price=price_2)
        cost_3 = cost_generator.generate_cost_curve("linear", strat_size, price=price_3)
        cost_4 = cost_generator.generate_cost_curve("linear", strat_size, price=price_4)

        game = pygambit.Game.new_table([strat_size] * 3)

        for strat_1 in range(1,strat_size):
            for strat_2 in range(1,strat_size):
                for strat_3 in range(1,strat_size):
                    #for strat_4 in range(1,strat_size):
                    strat_4 = 0
                    ct = strat_1 + strat_2 + strat_3 + strat_4
                    s_tput = tput[ct] if ct < len(tput) else tput[-1]
                    tput_1 = (strat_1 / ct) * s_tput if ct != 0 else 0
                    tput_2 = (strat_2 / ct) * s_tput if ct != 0 else 0
                    tput_3 = (strat_3 / ct) * s_tput if ct != 0 else 0
                    #tput_4 = (strat_4 / ct) * s_tput if ct != 0 else 0
                    game[strat_1, strat_2, strat_3][0] = round((tput_1 - cost_1[strat_1])*M)
                    game[strat_1, strat_2, strat_3][1] = round((tput_2 - cost_2[strat_2])*M)
                    game[strat_1, strat_2, strat_3][2] = round((tput_3 - cost_3[strat_3])*M)
                    #game[strat_1, strat_2, strat_3, strat_4][3] = round((tput_4 - cost_4[strat_4])*M)


        game.title = "Motivating Example"
        game.players[0].label = "Player 1"
        game.players[1].label = "Player 2"
        game.players[2].label = "Player 3"

        #pure_solver = pygambit.nash.ExternalEnumPureSolver()
        #pure_soln = pure_solver.solve(game)

        #if len(pure_soln) == 0:
        #    continue


        #solver = pygambit.nash.ExternalEnumPolySolver()
        solver = pygambit.nash.ExternalLogitSolver()
        soln = solver.solve(game)

        has_pure = False

        for i, psoln in enumerate(soln):
            #print(f"Solution {i}: {psoln}")
            total_profile = 0

            max_distance = 0 

            initial_strategies = [1] * n_players

            for k, pstrat in enumerate(psoln):
                player = k // (strat_size)
                n = (k % (strat_size)) + 1
                total_profile += n * pstrat
                initial_strategies[player] += n * pstrat
                max_distance = max(max_distance, abs(0.5 - pstrat) - 0.5)

            if max_distance < 0.1:
                #print("has pure")
                has_pure = True

            continuous_equilibria.append(get_continuous_equilibrium_degen([cost_1[1], cost_2[1], cost_3[1]], initial_strategies))
            mixed_equilibria.append(total_profile)

            print(f"continuous equilibrium: {continuous_equilibria[-1]}")
            print(f"mixed equilibrium: {mixed_equilibria[-1]}")

        print(f"equilibria_mape: {sum([abs(x-y)/x for x, y in zip(mixed_equilibria, continuous_equilibria)])/len(mixed_equilibria)}")
