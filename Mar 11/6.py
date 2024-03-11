# 6.How many people have the job title of "Lawyer" ? 
from dataframe import df
count_lawyers = df[df['Job']=='Lawyer'].shape[0]
print(f'The number of lawyers in the provided data is {count_lawyers}')