# 5.How many people have English 'en' as their Language of choice on the website? 
from dataframe import df 
count_en = df[df['Language']=='en'].shape[0]
print(count_en)