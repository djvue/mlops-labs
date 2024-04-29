import os
from catboost.datasets import titanic

titanic_train, titanic_test = titanic()

os.makedirs("data", exist_ok=True)
titanic_train.to_csv("data/titanic_train.csv", index=False)
titanic_test.to_csv("data/titanic_train.csv", index=False)