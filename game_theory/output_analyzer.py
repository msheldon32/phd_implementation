import pandas as pd
import pygambit

def get_tput_table(output_id, n_classes):
    header = [f"Class {i}" for i in range(n_classes)] + [f"Tput {i}" for i in range(n_classes)]
    df = pd.read_csv('toy_output_1/output_{}.csv'.format(output_id), header=None)
    df.columns = header
    return df

def get_tput(level, class_id, tput_table):
    for cid, clevel in enumerate(level):
        tput_table = tput_table[tput_table[f"Class {cid}"] == clevel]
    return float(tput_table[f"Tput {class_id}"].values[0])

def get_all_levels(possible_levels, n_classes):
    if n_classes == 1:
        return [[x] for x in possible_levels]
    else:
        return [[x] + y for x in possible_levels for y in get_all_levels(possible_levels, n_classes-1)]

if __name__ == "__main__":
    output_id = 7
    n_classes = 3

    prices = [3,3,3]
    possible_levels = [i for i in range(1, 11, 2)]


    all_levels = get_all_levels(possible_levels, n_classes)
    tput_table = get_tput_table(output_id, n_classes)

    game = pygambit.Game.new_table([len(possible_levels)] * n_classes)

    for level in all_levels:
        for i in range(n_classes):
            print(f"tput for class {i} at level {level} is {get_tput(level, i, tput_table)}")
            game[level][i] = get_tput(level, i, tput_table) - prices[i]*level[i]


