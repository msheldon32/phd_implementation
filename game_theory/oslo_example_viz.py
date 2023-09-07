import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    selected_player_counts = [1,3,5,9,11,13,15]
    markers = ['o','v','^','<','>','s','p','*','h']

    single_price_df = pd.read_csv("oslo_single_price.csv")

    single_price_df["price"] = single_price_df["price"].apply(lambda x: json.loads(x)[0])
    
    bp_data =[]
    for n_players in selected_player_counts:
        bp_data.append(single_price_df[single_price_df["n_players"] == n_players]["Q"])
    plt.boxplot(bp_data, labels=selected_player_counts)
    plt.xlabel("Number of players")
    plt.ylabel("Q")
    plt.show()


    bp_data =[]
    for n_players in selected_player_counts:
        bp_data.append(single_price_df[single_price_df["n_players"] == n_players]["eq_tput"])
    plt.boxplot(bp_data, labels=selected_player_counts)
    plt.xlabel("Number of players")
    plt.ylabel("T")
    plt.show()

    # plot Q vs price
    for n_players in selected_player_counts:
        plt.scatter(single_price_df[single_price_df["n_players"] == n_players]["price"],
                    single_price_df[single_price_df["n_players"] == n_players]["Q"],
                    label="n_players = {}".format(n_players),
                    alpha=0.1,
                    s=0.1,
                    marker=markers[selected_player_counts.index(n_players)])

    plt.xlabel("Price")
    plt.ylabel("Q")
    plt.legend()
    plt.show()

    #raise Exception()

    # plot T vs price
    for n_players in selected_player_counts:
        plt.scatter(single_price_df[single_price_df["n_players"] == n_players]["price"],
                    single_price_df[single_price_df["n_players"] == n_players]["eq_tput"],
                    label="n_players = {}".format(n_players),
                    alpha=0.1,
                    s=0.1,
                    marker=markers[selected_player_counts.index(n_players)])

    plt.xlabel("Price")
    plt.ylabel("T")
    plt.legend()
    plt.show()


    hetero_df = pd.read_csv("oslo_multi_price_het.csv")
    hetero_df["price"] = hetero_df["price"].apply(lambda x: json.loads(x))
    hetero_df["avg_price"] = hetero_df["price"].apply(lambda x: sum(x)/len(x))

    #hetero_df["price_ratio"] = hetero_df[["price", "avg_price"]].apply(lambda x: [p/x[1] for p in x[0]], axis=1)

    hetero_df["Qs_per_player"] = hetero_df["Qs_per_player"].apply(lambda x: json.loads(x))


    # plot Q vs price
    plt.scatter(hetero_df["avg_price"], hetero_df["Q"], alpha=0.05, marker=".")
    plt.scatter(single_price_df[single_price_df["n_players"] == 6]["price"],
                single_price_df[single_price_df["n_players"] == 6]["Q"],
                label="constant price",
                color="darkblue",
                alpha=0.1,
                s=0.1,
                marker=".")

    plt.xlabel("Avg Price")
    plt.ylabel("Q")
    plt.legend()
    plt.show()

    hetero_df = pd.read_csv("oslo_multi_price_het.csv")
    hetero_df["price"] = hetero_df["price"].apply(lambda x: json.loads(x))
    hetero_df["avg_price"] = hetero_df["price"].apply(lambda x: sum(x)/len(x))
    # plot T vs price
    plt.scatter(hetero_df["avg_price"], hetero_df["eq_tput"], alpha=0.05, marker=".")
    plt.scatter(single_price_df[single_price_df["n_players"] == 6]["price"],
                single_price_df[single_price_df["n_players"] == 6]["eq_tput"],
                label="constant price",
                color="darkblue",
                alpha=0.1,
                s=0.1,
                marker=".")
    plt.xlabel("Avg Price")
    plt.ylabel("T")
    plt.legend()
    plt.show()



    # collect all prices, Q per player Qs, Ts into a single list
    prices = []
    avg_prices = []
    Qs = []
    Qs_per_player = []
    Ts = []

    for i in range(len(hetero_df)):
        n_players = len(hetero_df.iloc[i]["price"])
        prices += hetero_df.iloc[i]["price"]
        avg_prices += [hetero_df.iloc[i]["avg_price"]]*len(hetero_df.iloc[i]["price"])
        Qs += [hetero_df.iloc[i]["Q"]]*len(hetero_df.iloc[i]["price"])
        Ts += [hetero_df.iloc[i]["eq_tput"]]*len(hetero_df.iloc[i]["price"])
        if type(hetero_df.iloc[i]["Qs_per_player"]) == str:
            Qs_per_player += json.loads(hetero_df.iloc[i]["Qs_per_player"])
        else:
            Qs_per_player += hetero_df.iloc[i]["Qs_per_player"]

        

    plt.scatter([p/avg_prices[i] for i,p in enumerate(prices)], [q/Qs[i] for i,q in enumerate(Qs_per_player)], alpha=0.05, marker=".")
    plt.xlabel("Price/Avg Price")
    plt.ylabel("$x_r$/Q")
    plt.legend()
    plt.show()

