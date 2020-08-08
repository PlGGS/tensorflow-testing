import pandas as pd

f = 0.2

df = pd.read_csv("cta/small_test_cta.csv")

test = df.sample(frac = f)
train = df.loc[~df.index.isin(test.index)]

test.to_csv(r'cta/tiny_test_cta.csv', index = False)
train.to_csv(r'cta/tiny_train_cta.csv', index = False)