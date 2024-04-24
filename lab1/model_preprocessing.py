from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURES = ["feature_1", "feature_2", "feature_3"]

if __name__ == '__main__':
    df_train = pd.read_csv("data/train/raw.csv")
    df_val = pd.read_csv("data/test/raw.csv")

    preprocessor = StandardScaler()

    df_train[FEATURES] = preprocessor.fit_transform(df_train[FEATURES])
    df_val[FEATURES] = preprocessor.fit_transform(df_val[FEATURES])

    df_train.to_csv("data/train/preprocessed.csv")
    df_val.to_csv("data/test/preprocessed.csv")
