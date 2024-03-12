# Q12:What are the top 5 most common jobs?
# A: Transit Operator                7036
# Special Nurse                   4389
# Registered Nurse                3736
# Public Svc Aide-Public Works    2518
# Police Officer 3                2421
from data_frame import df
print(df.JobTitle.value_counts().head(5))