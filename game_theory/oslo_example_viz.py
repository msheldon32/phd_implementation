import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    single_price_df = pd.read_csv("oslo_single_price.csv")

    single_price_df["price"] = single_price_df["price"].apply(lambda x: json.loads(x)[0])

    # plot Q vs number of players
    plt.scatter(single_price_df["n_players"], single_price_df["Q"])
    plt.xlabel("Number of players")
    plt.ylabel("Q")
    plt.show()

    # plot Q vs price
    plt.scatter(single_price_df["price"], single_price_df["Q"])
    plt.xlabel("Price")
    plt.ylabel("Q")
    plt.show()
