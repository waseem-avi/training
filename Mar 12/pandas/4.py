from data_frame import df
import pandas as pd
# Q4:What is the average BasePay ?
# A: 66325.44884050643
print(f"The average base salary is the mean of the base pay coloumn \n i.e : {df["BasePay"].mean()}")