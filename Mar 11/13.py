# 13.What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)
from dataframe import df 
# print(df['Email'])
top_5_email_providers = df['Email'].str.split("@").str[1].value_counts().index.tolist()[:5]
print(top_5_email_providers)