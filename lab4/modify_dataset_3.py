import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/titanic_train.csv")

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df[["Sex"]])

enc_df = pd.DataFrame(enc.transform(df[["Sex"]]).toarray(), columns=list(enc.categories_))

df[enc.categories_[0]] = enc_df[enc.categories_[0]]
df = df.rename(columns={"female": "Sex_female", "male": "Sex_male"}).drop(["Sex"], axis=1)

df = df.to_csv("data/titanic_train.csv", index=False)
