import requests
import pandas as pd
import scipy
import numpy as np
import sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    train_d=pd.read_csv(TRAIN_DATA_URL,index_col=False, header = None)
    test_d=pd.read_csv(TEST_DATA_URL, index_col=False, header = None)
    test_d = test_d.T
    train_d=train_d.T
    test_d.rename(columns={0: 'area', 1: 'price'},inplace=True)
    test_d= test_d.iloc[1:]
    train_d.rename(columns={0: 'area', 1: 'price'},inplace=True)
    train_d= train_d.iloc[1:]
    #print(train_d)
    #sns.regplot(x="area", y="price", data=train_d)
    lm = LinearRegression()
    lm
    X = train_d['area']
    Y = train_d['price']
    lm.fit(X,Y)
    #print(lm.score(X, Y))
    #print(lm.intercept_)
    #print(lm.coef_)
    area.reshape(1, -1)
    y_pred = lm.predict(area)
    return y_pred


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
