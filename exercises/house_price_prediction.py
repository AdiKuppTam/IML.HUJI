from re import M
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

# constants
DATA_PATH = "C:\\Users\\Adi\\OneDrive\\Documents\\Year4\\Sem2\\IML\\IML.HUJI\\datasets\\house_prices.csv"
PRICE_STR = "price"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=["id"])

    #df = df.dropna()
    df["date"] = df["date"].str.split("T").str[0]

    df = df[df["price"] > 0].fillna(0)

    df = pd.get_dummies(df,columns=["zipcode"], dtype=int, drop_first=True).astype(np.float64)

    df_just_data = df.drop(columns=["price"])

    return df_just_data, df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
       
    ln = LinearRegression(include_intercept=False)
    ln.fit(pd.DataFrame.to_numpy(X), y)
    y_pred = ln.predict(X.to_numpy())
    
    fig = go.Figure()
    for i in range(X.shape[1]):
        fig.add_trace(go.Scatter(x=X.iloc[:, i], y=y_pred, mode='markers', name=f"{X.columns[i]}", marker_color=i))
    fig.update_layout(title_text="Feature vs. Response", xaxis_title="Feature", yaxis_title="Response")
    fig.write_image(f"{output_path}/feature_evaluation.png")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, result = load_data(DATA_PATH)
    print(df.head())
    
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, result) 

    # Question 3 - Split samples into training- and testing sets.
    X_train, X_test, y_train, y_test = split_train_test(df, result)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    avg_loss_lst = []
    std_loss_lst = []
    for p in p_lst:
        X_train_p, y_train_p = X_train.sample(frac=p), y_train.sample(frac=p)
        ln = LinearRegression(include_intercept=True)
        ln.fit(X_train_p, y_train_p)
        y_pred = ln.predict(X_test)
        avg_loss_lst.append(np.mean(np.abs(y_pred - y_test)))
        std_loss_lst.append(4* np.std(np.abs(y_pred - y_test)))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_lst, y=avg_loss_lst,error_y=std_loss_lst, mode='lines', name="Average Loss"))
    fig.update_layout(title_text="Average Loss vs. Training Size", xaxis_title="Fraction of Training Size", yaxis_title="Average Loss")
    fig.write_image("avg_loss_vs_p.png")
    fig.show()



