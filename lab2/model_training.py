import pandas as pd
import pickle
import warnings

import xgboost as xgb

warnings.filterwarnings('ignore')

RANDOM_STATE = 324


def make_model():
    model_xgb = xgb.XGBRegressor(random_state=RANDOM_STATE,
                                 objective='reg:squarederror',
                                 enable_categorical=True,
                                 early_stopping_rounds=50,
                                 n_estimators=500)
    return model_xgb


if __name__ == '__main__':
    df_train = pd.read_csv("data/preprocessed_train.csv")

    model = make_model()

    X, y = df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]]

    model.fit(X, y, eval_set=[(X, y)], verbose=0)

    with open("data/model.pickle", "wb") as f:
        pickle.dump(model, f)
