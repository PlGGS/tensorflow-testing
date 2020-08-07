import pandas as pd

f = 0.2

df = pd.read_csv("cta/cta.csv")

test = df.sample(frac = f)
train = df.loc[~df.index.isin(test.index)]

test.to_csv(r'cta/test_cta.csv', index = False)
train.to_csv(r'cta/train_cta.csv', index = False)