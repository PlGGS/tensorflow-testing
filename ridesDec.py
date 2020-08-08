import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("cta/split_cta.csv")

vals = df[["rides"]].values
# max = df["rides"].max()
df[['rides']] = MinMaxScaler().fit_transform(vals)


df.to_csv(r'cta/split_scaled_cta.csv', index = False)