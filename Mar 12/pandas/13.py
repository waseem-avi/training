# Q13:How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)
# A: 202
from data_frame import df
print(sum(df[df['Year']==2013]['JobTitle'].value_counts()==1))