import os

import pandas as pd

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv('https://raw.githubusercontent.com/djvue/ml-models-mirror/main/house-prices-advanced-regression-techniques/train.csv')
    df.to_csv("data/data.csv", index=False)
