# 12.How many people have a credit card that expires in 2025? 
from dataframe import df
expiry_25 = df[df['CC Exp Date'].str.contains('/25')]
print(expiry_25.shape[0])
