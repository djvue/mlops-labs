import pandas as pd

df = pd.read_csv("data/titanic_train.csv")

df["Age"] = df["Age"].fillna(df["Age"].mean())

df = df.to_csv("data/titanic_train.csv", index=False)
