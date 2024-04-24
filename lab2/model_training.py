import pandas as pd
import pickle
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

RANDOM_STATE = 12


def make_model() -> LinearRegression:
    return LinearRegression()


if __name__ == '__main__':
    df_train = pd.read_csv("data/preprocessed_train.csv")

    model = make_model()

    X, y = df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]]

    model.fit(X, y)

    with open("data/model.pickle", "wb") as f:
        pickle.dump(model, f)
