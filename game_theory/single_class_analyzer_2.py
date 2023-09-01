import pandas as pd
from gt_analysis import TputData, CostGenerator

import random
import math
import time

STARTING_OUTPUT = 0
N_OUTPUTS = 10000
MIN_N_PLAYERS = 20
MAX_N_PLAYERS = 20
MIN_JOBS = 10
MAX_JOBS = 190
M = 100000

def check_equilibrium(prices, tput, total_ct, x_lo, x_hi, min_jobs, max_jobs):
    equilibrium_profile = [x for x in x_lo]
    total_eq_ct = sum(equilibrium_profile)

    n_players = len(prices)

    for player_id in range(n_players):
        if total_eq_ct == total_ct:
            break
        while equilibrium_profile[player_id] < x_hi[player_id] and total_eq_ct < total_ct:
            equilibrium_profile[player_id] += 1
            total_eq_ct += 1
    #print(f"equilibrium profile: {equilibrium_profile}, total_eq_ct: {total_eq_ct}, sum: {sum(equilibrium_profile)}")
    
    def get_z(job_ct, q):
        if (job_ct + q) >= len(tput):
            return tput[-1]/(job_ct + q)
        if job_ct + q == 0:
            return M
        return tput[job_ct + q]/(job_ct + q)

    for player_id in range(n_players):
        q = total_eq_ct - equilibrium_profile[player_id]

        eq_utility = (equilibrium_profile[player_id] * get_z(equilibrium_profile[player_id], q)) - (prices[player_id] * equilibrium_profile[player_id])

        for job_ct in range(min_jobs[player_id], max_jobs[player_id]+1):
            if job_ct == equilibrium_profile[player_id]:
                continue
            z = get_z(job_ct, q)
            utility = job_ct * z - prices[player_id] * job_ct

            if job_ct < equilibrium_profile[player_id]:
                if (utility >= eq_utility):
                    print(f"Player {player_id} is not in equilibrium, can increase util from {eq_utility} to {utility} by changing job count from {equilibrium_profile[player_id]} to {job_ct}")
                    return False
            elif utility > eq_utility:
                print(f"Player {player_id} is not in equilibrium, can increase util from {eq_utility} to {utility} by changing job count from {equilibrium_profile[player_id]} to {job_ct}")
                return False

    return True

def get_social_utility(prices, tput, total_ct, x_lo, x_hi, min_jobs, max_jobs):
    equilibrium_profile = [x for x in x_lo]
    total_eq_ct = sum(equilibrium_profile)

    n_players = len(prices)

    for player_id in range(n_players):
        if total_eq_ct == total_ct:
            break
        while equilibrium_profile[player_id] < x_hi[player_id] and total_eq_ct < total_ct:
            equilibrium_profile[player_id] += 1
            total_eq_ct += 1
    
    def get_z(job_ct, q):
        if (job_ct + q) >= len(tput):
            return tput[-1]/(job_ct + q)
        if job_ct + q == 0:
            return M
        return tput[job_ct + q]/(job_ct + q)

    for player_id in range(n_players):
        q = total_eq_ct - equilibrium_profile[player_id]

        eq_utility = (equilibrium_profile[player_id] * get_z(equilibrium_profile[player_id], q)) - (prices[player_id] * equilibrium_profile[player_id])

        for job_ct in range(min_jobs[player_id], max_jobs[player_id]+1):
            if job_ct == equilibrium_profile[player_id]:
                continue
            z = get_z(job_ct, q)
            utility = job_ct * z - prices[player_id] * job_ct

    return True

if __name__ == "__main__":
    tput_data = TputData()
    cost_generator = CostGenerator()

    starting_time = time.time()

    output_file = f"{sys.argv[1]}/{sys.argv[2]}.csv"

    with open(output_file, "w") as f:
        f.write("output_id,n_players,prices,min_acc,max_acc,social_optimum_index,soc_stability,soc_anarchy,soc_optimum,min_tput,max_tput,opt_tput\n")

    for output_id in range(STARTING_OUTPUT, STARTING_OUTPUT+N_OUTPUTS):
        if output_id % 100 == 0:
            print(f"Output {output_id}")
        found_eq = False
        n_eq = 0
        n_players = random.randint(MIN_N_PLAYERS, MAX_N_PLAYERS)

        cost_curves = []
        prices = []

        for player_id in range(n_players):
            shape = random.choice(['linear'])#,'convex', 'exp'])
            price = random.random()**3
            cost_curve = cost_generator.generate_cost_curve(shape, tput_data.n_jobs+1, price)
            cost_curves.append(cost_curve)

            prices.append(cost_curve[1])


        min_jobs = [MIN_JOBS for _ in range(n_players)]
        max_jobs = [MAX_JOBS for _ in range(n_players)]
        max_resid_jobs = [sum(max_jobs)-max_jobs[r] for r in range(n_players)]
        min_resid_jobs = [sum(min_jobs)-min_jobs[r] for r in range(n_players)]

        social_optimum_amount = float("-inf")
        social_optimum_index = -1

        min_equilibrium_amount = float("inf")
        equilibrium_index = -1
        max_equilibrium_amount = float("-inf")

        sorted_prices = sorted([(price, player_id) for player_id, price in enumerate(prices)])

        def get_cumulative_so_price(total_jobs):
            # calculate up to minimum jobs
            jobs_per_player = [min_jobs[sorted_prices[i][1]] for i in range(n_players)]

            min_total_jobs = sum(jobs_per_player)

            if min_total_jobs > total_jobs:
                return float("-inf")
            
            min_player_idx = 0
            while min_total_jobs < total_jobs:
                if jobs_per_player[min_player_idx] >= max_jobs[sorted_prices[min_player_idx][1]]:
                    min_player_idx += 1
                    continue
                jobs_per_player[min_player_idx] += 1
                min_total_jobs += 1
            return sum([jobs_per_player[i] * sorted_prices[i][0] for i in range(n_players)])

            
        # loop through each job count and find pivots
        min_pivots = [float("inf") for _ in range(n_players)]
        x_lo = [0 for _ in range(n_players)]
        x_hi = [0 for _ in range(n_players)]

        prev_max_x = [-1 for _ in range(n_players)]
        for next_job_ct in range(1, sum(max_jobs)+2):
            is_valid = True

            Q = next_job_ct - 1
            
            if next_job_ct > tput_data.n_jobs:
                z = tput_data.run_data[output_id][-1]/next_job_ct
            else:
                z = tput_data.run_data[output_id][next_job_ct]/next_job_ct
            if next_job_ct == 1:
                delta_z = 0
            else:
                if next_job_ct-1 > tput_data.n_jobs:
                    delta_z = z - (tput_data.run_data[output_id][-1]/(next_job_ct-1))
                else:
                    delta_z = z - (tput_data.run_data[output_id][next_job_ct-1]/(next_job_ct-1))

            t = (z-delta_z)*Q
            social_utility = t - get_cumulative_so_price(next_job_ct)

            if social_utility > social_optimum_amount:
                social_optimum_amount = social_utility
                social_optimum_index = next_job_ct

            for player_id in range(n_players):
                i1_min = max(min_jobs[player_id], Q-max_resid_jobs[player_id])
                i1_max = min(max_jobs[player_id], Q-min_resid_jobs[player_id])

                if i1_min > i1_max:
                    is_valid = False
                    continue

                delta_u = lambda x: (x * delta_z) + z - prices[player_id]
                in_i1 = lambda x: i1_min <= x <= i1_max

                if delta_z == 0:
                    if prices[player_id] >= z:
                        i2_min = i1_min
                    else:
                        is_valid = False
                        continue
                else:
                    est_pivot = (prices[player_id] - z)/delta_z
                    pivot_floor = math.floor(est_pivot)
                    pivot_ceil = math.ceil(est_pivot)

                    if in_i1(pivot_floor) and (delta_u(pivot_floor) <= 0):
                        i2_min = pivot_floor
                    elif in_i1(pivot_ceil):
                        i2_min = pivot_ceil
                    elif pivot_floor >= i1_max and (max_jobs[player_id] == i1_max):
                        i2_min = max_jobs[player_id]
                    elif pivot_ceil <= i1_min:
                        i2_min = i1_min
                    else:
                        is_valid = False
                        continue

                i2_max = i1_max

                if i2_min > i2_max:
                    is_valid = False
                    continue

                i3_min = i2_min
                i3_max = min(min_pivots[player_id] + Q - 1, i2_max)

                min_pivots[player_id] = min(min_pivots[player_id], i2_min-Q)

                if i3_min > i3_max:
                    is_valid = False
                    continue

                x_lo[player_id] = i3_min
                x_hi[player_id] = i3_max
                    

            if is_valid and sum(x_lo) <= (next_job_ct-1) <= sum(x_hi):
                found_eq = True
                n_eq += 1

                equilibrium_index = Q

                assert check_equilibrium(prices, tput_data.run_data[output_id], next_job_ct-1, x_lo, x_hi, min_jobs, max_jobs)

                if next_job_ct-1 != 0:
                    break
            elif not is_valid:
                pass
            else:
                pass
        
        assert n_eq == 1
        with open("output.txt", "a") as f:
            f.write(f"{output_id},{n_players},{prices},{equilibrium_index},{equilibrium_index},{social_optimum_index},")
    ending_time = time.time()

    print(f"found equilibria in {ending_time-starting_time} seconds")

