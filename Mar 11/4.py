# 4.What were the highest and lowest purchase prices?
from dataframe import df 
min = df['Purchase Price'].min()
max = df['Purchase Price'].max()
print(f'The minimum purchase price was {min}')
print(f'The maximum purchase price was {max}')

