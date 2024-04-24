from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from joblib import load

FEATURES = ["feature_1", "feature_2", "feature_3"]

if __name__ == '__main__':
    model = load("data/model.joblib")
    df_val = pd.read_csv("data/test/raw.csv")

    y_pred = model.predict(df_val[FEATURES])

    metrics = {
        "mse": mean_squared_error(df_val["y"], y_pred),
        "r2": r2_score(df_val["y"], y_pred),
    }

    print("model metrics:", metrics, sep="\n")
