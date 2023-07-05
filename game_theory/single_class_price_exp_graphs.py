import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

if __name__ == "__main__":
    price_data = pd.DataFrame()

    for run_no in range(1,7):
        filename = f"price_exp/run1_{run_no}.csv"
        price_data = pd.concat([price_data, pd.read_csv(filename)])
    price_data["price_of_anarchy"] = price_data["soc_optimum"] / price_data["soc_anarchy"]
    #price_data["price_of_stability"] = price_data["soc_optimum"] / price_data["soc_stability"]

    price_data["price2"] = price_data["price"] ** 2
    price_data["price3"] = price_data["price"] ** 3
    price_data["price4"] = price_data["price"] ** 4
    price_data["price5"] = price_data["price"] ** 5
    price_data["price6"] = price_data["price"] ** 6

    # create a bar chart over n_classes
    #plt.bar(price_data["n_classes"], price_data["price_of_anarchy"], label="Price of Anarchy")
    #plt.bar(price_data["n_classes"], price_data["price_of_stability"], label="Price of Stability")
    #plt.xlabel("Number of Players")
    #plt.ylabel("Price of Stability")
    #plt.title("Price of Stability over Number of Players")
    #plt.legend()
    #plt.show()


    for n_classes in range(2, 5):
        n_classes_data = price_data[price_data["n_classes"] == n_classes].sort_values(by="price")
        plt.scatter(price_data["price"], price_data["price_of_anarchy"], label="Price of Anarchy")
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
        plt.ylabel("Price of Stability")
        plt.title(f"Price of Stability over Price for {n_classes} Players")
        plt.legend()
        plt.show()

        plt.scatter(n_classes_data["price"], n_classes_data["min_acc"], label="Number of Vehicles at Equilibrium")
        plt.scatter(n_classes_data["price"], n_classes_data["social_optimum_index"], label="Number of Vehicles at Optimum")
        plt.xlabel("Price")
        plt.ylabel("Number of Vehicles at Equilibrium vs Social Optimum")
        plt.title(f"Number of Vehicles at Equilibrium vs Social Optimum over Price for {n_classes} Players")
        plt.legend()
        plt.show()
