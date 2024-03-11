import pandas as pd

filepath = 'Mar 11/data.txt'
df = pd.read_csv(filepath)
if __name__ == '__main':
    print(df.columns)
