import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


def train_model(X_tr, y_tr):
    model = LinearRegression()  # Chosen model - Linear Regression
    model.fit(X_tr, y_tr)  # Fit model using train datasets
    return model


def data_split(X_, y_):
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test


def prediction_and_mape(X_, y_):
    mape_array = list()
    for i in range(2, 5):
        X_train, X_test, y_train, y_test = data_split(X_**i, y_)  # Split predictor and data and making predictor squared
        model = train_model(X_train, y_train)
        prediction_salary_squared_pred = model.predict(X_test)  # predict target (salary) based on player's rating
        mape_test_squared = mape(y_test, prediction_salary_squared_pred)
        mape_array.append(mape_test_squared)
    return mape_array


def calculating_mape(X_, y_):
    X_train, X_test, y_train, y_test = data_split(X_, y_)
    model = train_model(X_train, y_train)
    predicted_salary = model.predict(X_test)
    mape_test = mape(y_test, predicted_salary)
    return mape_test


if __name__ == "__main__":
    # checking ../Data directory presence
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # download data if it is unavailable
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)

    # read data
    data = pd.read_csv('../Data/data.csv')

    # write your code here

    # First stage
    '''
    X, y = data[["rating"]], data["salary"]  # Choose predictor and target
    '''

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)  # Split predictor and data

    model = LinearRegression()  # Chosen model - Linear Regression
    model.fit(X_train, y_train)  # Fit model using train datasets
    prediction_salary = model.predict(X_test)  # predict target (salary) based on player's rating (that's why we use "X_test" in model.predict(...))
    mape_test = mape(y_test, prediction_salary)  # Calculating the mean_absolute_percentage_error

    print(round(model.intercept_, 5), round(float(model.coef_), 5), round(mape_test, 5))  # print the results rounded to 5 digits after the dot
    '''

    # Second Stage
    '''
    mapes_array = prediction_and_mape(X, y)
    print(round(min(mapes_array), 5))
    '''

    # Third Stage
    '''
    X, y = data.drop(columns="salary"), data["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = train_model(X_train, y_train)
    print(*model.coef_, sep=", ")
    '''

    # Fourth Stage

    '''
    X, y = data.drop(columns="salary"), data["salary"]
    # print(data.corr())
    mape_array = list()
    mape_array.append(calculating_mape(X.drop(columns="age"), y))
    mape_array.append(calculating_mape(X.drop(columns="rating"), y))
    mape_array.append(calculating_mape(X.drop(columns="experience"), y))
    mape_array.append(calculating_mape(X.drop(columns=["age", "rating"]), y))
    mape_array.append(calculating_mape(X.drop(columns=["age", "experience"]), y))  # the best one
    mape_array.append(calculating_mape(X.drop(columns=["rating", "experience"]), y))
    print(mape_array)
    print(round(min(mape_array), 5))
    # rating/salary/experience
    '''

    # Fifth Stage
    X, y = data.drop(columns=["salary", "age", "experience"]), data["salary"]
    X_train, X_test, y_train, y_test = data_split(X, y)
    model = train_model(X_train, y_train)
    predicted_salary = model.predict(X_test)

    first_way = predicted_salary.copy()
    first_way[first_way < 0] = 0

    y_train_median = y_train.median()
    second_way = predicted_salary.copy()
    second_way[second_way < 0] = y_train_median

    print(round(min([mape(y_test, first_way), mape(y_test, second_way)]), 5))

    # print(predicted_salary, type(predicted_salary))
