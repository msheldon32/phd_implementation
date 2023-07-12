import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

if __name__ == "__main__":
    price_data = pd.DataFrame()

    for run_no in range(1,7):
        filename = f"price_exp_2/run1_{run_no}.csv"
        price_data = pd.concat([price_data, pd.read_csv(filename)])
    print(f"columns: {price_data.columns}")
    price_data["price_of_anarchy"] = price_data["soc_optimum"] / price_data["soc_anarchy"]
    #price_data["price_of_stability"] = price_data["soc_optimum"] / price_data["soc_stability"]

    price_data["price2"] = price_data["price"] ** 2
    price_data["price3"] = price_data["price"] ** 3
    price_data["price4"] = price_data["price"] ** 4
    price_data["price5"] = price_data["price"] ** 5
    price_data["price6"] = price_data["price"] ** 6

    # create a bar chart over n_classes
    price_of_anarchy_per_class = price_data.groupby("n_classes").mean()["price_of_anarchy"]
    print(f"price_of_anarchy_per_class: {price_of_anarchy_per_class}")
    plt.plot(price_of_anarchy_per_class.index, price_of_anarchy_per_class, marker="o")
    #plt.bar(price_data["n_classes"], price_data["price_of_stability"], label="Price of Stability", alpha=0.01)
    plt.xlabel("Number of Players")
    plt.ylabel("Price of Anarchy")
    plt.xticks(price_of_anarchy_per_class.index)
    plt.xlim(1.5, 4.5)
    plt.ylim([1, price_of_anarchy_per_class.max() + 0.1])
    #plt.title("Price of Anarchy over Number of Players")
    plt.legend()
    plt.show()

    number_of_vehicles_per_class = price_data.groupby("n_classes").mean()["min_acc"]
    opt_number_of_vehicles_per_class = price_data.groupby("n_classes").mean()["social_optimum_index"]
    print(f"vehicles per class: {number_of_vehicles_per_class}")
    print(f"opt vehicles per class: {opt_number_of_vehicles_per_class}")
    plt.plot(number_of_vehicles_per_class.index, number_of_vehicles_per_class, marker="o", label="Number of Vehicles at Equilibrium")
    plt.plot(opt_number_of_vehicles_per_class.index, opt_number_of_vehicles_per_class, marker="o", label="Number of Vehicles at Social Optimum")
    plt.xlabel("Number of Players")
    plt.ylabel("Number of Vehicles")
    plt.xticks(number_of_vehicles_per_class.index)
    plt.xlim(1.5, 4.5)
    plt.ylim([0, number_of_vehicles_per_class.max() + 5])
    #plt.title("Number of Vehicles at Equilibrium and Social Optimum")
    plt.legend()
    plt.show()

    trips_per_class = price_data.groupby("n_classes").mean()[" min_tput"]
    opt_trips_per_class = price_data.groupby("n_classes").mean()[" opt_tput"]
    print(f"trips per class: {trips_per_class}")
    print(f"opt trips per class: {opt_trips_per_class}")
    plt.plot(trips_per_class.index, trips_per_class, marker="o", label="Number of Trips at Equilibrium")
    plt.plot(opt_trips_per_class.index, opt_trips_per_class, marker="o", label="Number of Trips at Social Optimum")
    plt.xlabel("Number of Players")
    plt.ylabel("Number of Trips")
    plt.xticks(trips_per_class.index)
    plt.xlim(1.5, 4.5)
    plt.ylim([0, trips_per_class.max() + 5])
    #plt.title("Number of Trips at Equilibrium and Social Optimum")
    plt.legend()
    plt.show()


    for n_classes in range(2, 5):
        n_classes_data = price_data[price_data["n_classes"] == n_classes].sort_values(by="price")
        n_classes_data["bin_price"] = pd.cut(n_classes_data["price"], bins=10).apply(lambda x: x.mid)
        price_of_anarchy_per_price = n_classes_data.groupby("bin_price").mean()["price_of_anarchy"]
        #plt.scatter(price_data["price"], price_data["price_of_anarchy"], label="Price of Anarchy")
        plt.plot(price_of_anarchy_per_price.index, price_of_anarchy_per_price, label="Price of Anarchy", marker="o")
        plt.xlim([0, 5])
        plt.ylim([1, price_of_anarchy_per_price.max() + 0.1])
        #plt.scatter(n_classes_data["price"], n_classes_data["price_of_stability"], label="Price of Stability")

        # regress over price
        #reg = LinearRegression().fit(n_classes_data[["price"]],n_classes_data["price_of_stability"].apply(lambda x: math.log(x)))
        
        #sm_reg = sm.OLS(n_classes_data["price_of_stability"].apply(lambda x: math.log(x)), n_classes_data[["price"]]).fit()
        #print(sm_reg.summary())

        # create q q plot
        #logit_pos = n_classes_data["price_of_stability"].apply(lambda x: math.log((x-0.9)/(1-(x-0.9))))
        #sm.qqplot(n_classes_data["price_of_stability"], line='45')
        #sm.qqplot(logit_pos, line='45')
        #plt.show()

        # plot the regression line
        #plt.plot(n_classes_data["price"], pd.Series(reg.predict(n_classes_data[["price"]])).apply(lambda x: math.exp(x)), color="red", label="Regression Line")

        plt.xlabel("Price")
        plt.ylabel("Price of Anarchy")
        #plt.title(f"Price of Anarchy over Price for {n_classes} Players")
        plt.legend()
        plt.show()

        number_of_vehicles_per_price = n_classes_data.groupby("bin_price").mean()["min_acc"]
        opt_number_of_vehicles_per_price = n_classes_data.groupby("bin_price").mean()["social_optimum_index"]
        plt.plot(number_of_vehicles_per_price.index, number_of_vehicles_per_price, label="Number of Vehicles at Equilibrium", marker="o")
        plt.plot(opt_number_of_vehicles_per_price.index, opt_number_of_vehicles_per_price, label="Optimal Number of Vehicles", marker="o")
        plt.xlim([0, 5])
        plt.ylim([0, number_of_vehicles_per_price.max() + 5])
        plt.xlabel("Price")
        plt.ylabel("Number of Vehicles")
        #plt.title(f"Number of Vehicles over Price for {n_classes} Players")
        plt.legend()
        plt.show()

        trips_per_price = n_classes_data.groupby("bin_price").mean()[" min_tput"]
        opt_trips_per_price = n_classes_data.groupby("bin_price").mean()[" opt_tput"]
        plt.plot(trips_per_price.index, trips_per_price, label="Number of Trips at Equilibrium", marker="o")
        plt.plot(opt_trips_per_price.index, opt_trips_per_price, label="Optimal Number of Trips", marker="o")
        plt.xlim([0, 5])
        plt.ylim([0, trips_per_price.max() + 5])
        plt.xlabel("Price")
        plt.ylabel("Number of Trips")
        #plt.title(f"Number of Trips over Price for {n_classes} Players")
        plt.legend()
        plt.show()
