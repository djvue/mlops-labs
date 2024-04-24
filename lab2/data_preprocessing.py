import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from columns import NUM_FEATURES, CAT_FEATURES, Y_COLUMN, ALL_FEATURES


warnings.filterwarnings('ignore')


RANDOM_STATE = 234

num_pipe_scaler_columns = ['LotFrontage', 'OverallQual', 'OverallCond', 'GrLivArea', '1stFlrSF', 'BedroomAbvGr',
                           'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
num_pipe_power_columns = [el for el in NUM_FEATURES if el not in [*num_pipe_scaler_columns]]
cat_pipe_ordinal_columns = ['PavedDrive', 'CentralAir']
cat_pipe_rare_one_hot_columns = [el for el in CAT_FEATURES if el not in cat_pipe_ordinal_columns]


class RareGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, other_value='Other'):
        self.threshold = threshold
        self.other_value = other_value
        self.freq_dict = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            freq = X[col].value_counts(normalize=True)
            self.freq_dict[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X.select_dtypes(include=['object']):
            X_copy[col] = X_copy[col].apply(
                lambda x: x if x in self.freq_dict[col] or x is np.nan else self.other_value)
        return X_copy


def make_preprocessors() -> ColumnTransformer:
    num_pipe_scaler = Pipeline([
        ('imput', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ]) # SalePrice

    num_pipe_power = Pipeline([
        ('imput', SimpleImputer(strategy='mean')),
        ('power', PowerTransformer()),
    ])

    cat_pipe_ordinal = Pipeline([
        ('encoder', OrdinalEncoder()),
    ])
    cat_pipe_ordinal_columns = ['PavedDrive', 'CentralAir']

    cat_pipe_rare_one_hot = Pipeline([
        ('replace_rare', RareGrouper(threshold=0.003, other_value='Other')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessors = ColumnTransformer(transformers=[
        ('num_scaler', num_pipe_scaler, num_pipe_scaler_columns),
        ('num_power', num_pipe_power, num_pipe_power_columns),
        ('cat_ordinal', cat_pipe_ordinal, cat_pipe_ordinal_columns),
        ('cat_rare_one_hot', cat_pipe_rare_one_hot, cat_pipe_rare_one_hot_columns),
    ])

    return preprocessors


def run_preprocess():
    df = pd.read_csv("data/train.csv")

    df = df[df[Y_COLUMN].notna()]

    X_train, X_val, y_train, y_val = train_test_split(df[ALL_FEATURES], df[Y_COLUMN],
                                                      test_size=0.3,
                                                      random_state=RANDOM_STATE)

    X_train, X_val = X_train.reset_index(), X_val.reset_index(),
    y_train, y_val = pd.Series(y_train).reset_index(drop=True), pd.Series(y_val).reset_index(drop=True)

    preprocessors = make_preprocessors()

    preprocessor_fitted = preprocessors.fit(X_train)
    preprocessor_fitted = preprocessor_fitted.fit(X_val)

    cat_pipe_rare_one_hot_names = preprocessors.transformers_[3][1]['encoder'].get_feature_names_out(cat_pipe_rare_one_hot_columns)
    columns = np.hstack([
        num_pipe_scaler_columns,
        num_pipe_power_columns,
        cat_pipe_ordinal_columns,
        cat_pipe_rare_one_hot_names,
    ])

    X_preprocessed_train = preprocessor_fitted.transform(X_train)
    X_preprocessed_val = preprocessor_fitted.transform(X_val)

    df_preprocessed_train = pd.DataFrame(X_preprocessed_train, columns=columns)
    df_preprocessed_val = pd.DataFrame(X_preprocessed_val, columns=columns)

    df_preprocessed_train[Y_COLUMN] = y_train
    df_preprocessed_val[Y_COLUMN] = y_val

    df_preprocessed_train.to_csv("data/preprocessed_train.csv", index=False)
    df_preprocessed_val.to_csv("data/preprocessed_test.csv", index=False)


if __name__ == '__main__':
    run_preprocess()
