# Q11: How many unique job titles are there?
# A:2159

from data_frame import df
print(df['JobTitle'].nunique())