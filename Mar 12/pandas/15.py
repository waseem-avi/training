# Q15: Bonus: Is there a correlation between length of the Job Title string and Salary? 
# A: 
# 			TotalPayBenefits	title_len
# TotalPayBenefits	1.000000		-0.036878
# title_len		-0.036878		1.000000
from data_frame import df

df['titlelen']=df['JobTitle'].apply(len)
print(df[['titlelen','TotalPayBenefits']].corr())