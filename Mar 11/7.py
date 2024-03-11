# 7.How many people made the purchase during the AM and how many people made the purchase during PM ? 
from dataframe import df 
# print(df['AM or PM'])
count_am = df[df['AM or PM']=='AM'].shape[0]
count_pm = df[df['AM or PM']=='PM'].shape[0]
print(f'Purcahses made during AM : {count_am}')
print(f'Purcahses made during PM : {count_pm}')

