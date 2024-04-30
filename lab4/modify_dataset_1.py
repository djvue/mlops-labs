import pandas as pd

df = pd.read_csv("data/titanic_train.csv")

df = df.drop([1, 2, 3])

df = df.to_csv("data/titanic_train.csv", index=False)
