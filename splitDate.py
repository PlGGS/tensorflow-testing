import pandas as pd

df = pd.read_csv("cta/cta.csv")
df[['month', 'day', 'year']] = df.date.str.split('/', expand=True)
df.to_csv(r'cta/split_cta.csv', index = False)