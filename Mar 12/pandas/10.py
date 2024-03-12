# Q10:What was the average (mean) BasePay of all employees per year? (2011-2014) ? 
# A:  Year
# 2011    63595.956517
# 2012    65436.406857
# 2013    69630.030216
# 2014    66564.421924
# Name: BasePay, dtype: float64
from data_frame import df
x=df.groupby('Year')
# print(type(df.Year))
print(x['BasePay'].mean())
