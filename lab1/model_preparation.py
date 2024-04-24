from sklearn.linear_model import LinearRegression
import pandas as pd
from joblib import dump

FEATURES = ["feature_1", "feature_2", "feature_3"]

if __name__ == '__main__':
    df_train = pd.read_csv("data/train/raw.csv")

    model = LinearRegression()

    model.fit(df_train[FEATURES], df_train["y"])

    dump(model, "data/model.joblib")
