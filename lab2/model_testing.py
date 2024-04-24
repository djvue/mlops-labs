import pandas as pd
import pickle
import warnings

from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

RANDOM_STATE = 234

if __name__ == '__main__':
    with open("data/model.pickle", "rb") as f:
        model = pickle.load(f)

    df_val = pd.read_csv("data/preprocessed_test.csv")

    X, y = df_val[df_val.columns[:-1]], df_val[df_val.columns[-1]]

    y_pred = model.predict(X)

    metrics = {
        "mse": mean_squared_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }

    print("model metrics:", metrics, sep="\n")
