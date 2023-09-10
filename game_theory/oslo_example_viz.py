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
        df = single_price_df[single_price_df["n_players"] == n_players].sort_values("price")
        plt.plot(df["price"],
                    df["Q"],
                    label="n_players = {}".format(n_players),
                    markevery=1500,
                    marker=markers[selected_player_counts.index(n_players)],
                    fillstyle='none',
                    )
                    #alpha=0.1,
                    #s=0.1,
                    #marker=markers[selected_player_counts.index(n_players)])

    plt.xlabel("Price")
    plt.ylabel("Q")
    plt.legend()
    plt.show()

    #raise Exception()

    # plot T vs price
    for n_players in selected_player_counts:
        df = single_price_df[single_price_df["n_players"] == n_players].sort_values("price")
        plt.plot(df["price"],
                    df["eq_tput"],
                    label="n_players = {}".format(n_players),
                    markevery=1500,
                    marker=markers[selected_player_counts.index(n_players)],
                    fillstyle='none',
                 )
                    #alpha=0.1,
                    #s=0.1,
                    #marker=markers[selected_player_counts.index(n_players)])

    plt.xlabel("Price")
    plt.ylabel("T")
    plt.legend()
    plt.show()


    hetero_df = pd.read_csv("oslo_multi_price_het.csv")
    hetero_df["price"] = hetero_df["price"].apply(lambda x: json.loads(x))
    hetero_df["avg_price"] = hetero_df["price"].apply(lambda x: round(sum(x)/len(x),2))

    #hetero_df["price_ratio"] = hetero_df[["price", "avg_price"]].apply(lambda x: [p/x[1] for p in x[0]], axis=1)

    hetero_df["Qs_per_player"] = hetero_df["Qs_per_player"].apply(lambda x: json.loads(x))

    hetero_df = hetero_df.sort_values("avg_price")

    single_price_6 = single_price_df[single_price_df["n_players"] == 6].sort_values("price")

    # plot Q vs price
    plt.plot(hetero_df.groupby("avg_price").mean().index,
            hetero_df.groupby("avg_price").mean()["Q"],
             marker="x",
             label="heterogeneous price",
             markevery=8)
    plt.fill_between(hetero_df.groupby("avg_price").mean().index,
            hetero_df.groupby("avg_price").mean()["Q"] - hetero_df.groupby("avg_price").std()["Q"],
            hetero_df.groupby("avg_price").mean()["Q"] + hetero_df.groupby("avg_price").std()["Q"],
            alpha=0.2)
    plt.plot(single_price_6["price"],
                single_price_6["Q"],
                label="constant price",
                marker="o",
                fillstyle="none",
                markevery=1500)


    plt.xlabel("Avg Price")
    plt.ylabel("Q")
    plt.legend()
    plt.show()

    # plot T vs price
    plt.plot(hetero_df.groupby("avg_price").mean().index,
            hetero_df.groupby("avg_price").mean()["eq_tput"],
             marker="x",
             label="heterogeneous price",
             markevery=8)
    plt.fill_between(hetero_df.groupby("avg_price").mean().index,
            hetero_df.groupby("avg_price").mean()["eq_tput"] - hetero_df.groupby("avg_price").std()["eq_tput"],
            hetero_df.groupby("avg_price").mean()["eq_tput"] + hetero_df.groupby("avg_price").std()["eq_tput"],
            alpha=0.2)
    plt.plot(single_price_6["price"],
                single_price_6["eq_tput"],
                label="constant price",
                marker="o",
                fillstyle="none",
                markevery=1500)


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


    print(f"max Q: {max(Qs)}")
        

    plt.scatter([p/avg_prices[i] for i,p in enumerate(prices)], [q/Qs[i] for i,q in enumerate(Qs_per_player)], alpha=0.05, marker=".")
    plt.xlabel("Price/Avg Price")
    plt.ylabel("$x_r$/Q")
    plt.legend()
    plt.show()

