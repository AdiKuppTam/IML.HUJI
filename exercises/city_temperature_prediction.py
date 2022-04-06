# from os import plock
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"

# constats, so we won't have an annoying errors :)
COUNTRY_STR = "Country"
TEMP_STR = "Temp"
MOUNTH_STR = "Month"
DAY_OF_YEAR_STR = "DayOfYear"
DATE_STR = "Date"
TEMP_MEAN_STR = "TempMean"
TEMP_STD_STR = "TempStd"



def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename)


    # calculating the exact day in year, so we can compare the data
    df[DAY_OF_YEAR_STR] =  pd.to_datetime(df[DATE_STR]).dt.dayofyear

    # returning the dataframe, without corner (impossible) values
    return df[df[TEMP_STR] > -60]


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:\\Users\\Adi\\OneDrive\\Documents\\Year4\\Sem2\\IML\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    just_israel = df[df[COUNTRY_STR] == "Israel"]

    #seif alef
    just_israel.plot(kind="scatter", x=DAY_OF_YEAR_STR, y=TEMP_STR, c="Year", colormap='Accent')
    
    # seif beit
    just_israel.groupby(MOUNTH_STR).Temp.mean().plot(kind="bar", yerr=just_israel.groupby(MOUNTH_STR).Temp.std(), colormap='Accent')

    # df = px.data.gapminder().query(COUNTRY_STR)

    # Question 3 - Exploring differences between countries
    #df[TEMP_MEAN_STR] = df.groupby([COUNTRY_STR, MOUNTH_STR]).mean()
    #df[TEMP_STD_STR] = df.groupby([COUNTRY_STR, MOUNTH_STR]).std()
#
    #df[TEMP_MEAN_STR].plot(x=MOUNTH_STR, kind="bar", yerr=df[TEMP_STD_STR], colormap='Accent')

    month_range = np.arange(1,13)

    for country in df[COUNTRY_STR].unique():
        country_df = df[df[COUNTRY_STR] == country]
        month_std = country_df.groupby(MOUNTH_STR).std()
        month_avg = country_df.groupby(MOUNTH_STR).mean()
        plt.errorbar(x=month_range,y=month_avg[TEMP_STR],yerr = month_std[TEMP_STR], label=country)
    plt.legend()
    plt.show()


    # Question 4 - Fitting model for different values of `k`
    to_np = lambda x: x.to_numpy().flatten()
    data_train, data_test, target_train, target_test = to_np(split_train_test(just_israel[DAY_OF_YEAR_STR], just_israel[TEMP_STR], train_proportion=0.75))
    loss_arr = []
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(data_train, target_train)
        loss_arr.append(poly.loss(data_test, target_test))
        print(f"k={k} R2={poly.score(df.DayInYear, df.Temp)}")  # R2 is the coefficient of determination

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()