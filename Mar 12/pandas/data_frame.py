import pandas as pd
df = pd.read_csv("Mar 12/pandas/Salaries.csv")
if __name__=="__main":
    print(df.head(5))
    print(df.info())