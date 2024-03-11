# 9.Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction?
from dataframe import df
print(df['Lot'])
row = df[df['Lot']=='90 WT']
price = row['Purchase Price']
print(price.values)