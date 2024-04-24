import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 345
FEATURES = ["feature_1", "feature_2", "feature_3"]

if __name__ == '__main__':
    X, y = make_regression(n_samples=5000, n_features=3, noise=1, random_state=RANDOM_STATE)

    X = pd.DataFrame(X, columns=FEATURES)
    y = pd.Series(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=RANDOM_STATE)

    df_train = X_train.copy()
    df_train["y"] = y_train

    df_val = X_val.copy()
    df_val["y"] = y_val

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    df_train.to_csv("data/train/raw.csv", index=False)
    df_val.to_csv("data/test/raw.csv", index=False)
