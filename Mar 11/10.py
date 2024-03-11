# 10.What is the email of the person with the following Credit Card Number: 4926535242672853
from dataframe import df
row = df[df['Credit Card']==4926535242672853]
# print(row)
value = row['Email']
print(value.values)