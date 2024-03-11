# 8.What are the 5 most common Job Titles?
from dataframe import df 
top_5_jobs = df['Job'].value_counts().index.tolist()[:5]
print(top_5_jobs)

# df['Job'].value_counts(): This part of the code computes the frequency count of unique values in the 'Job' column of the DataFrame df. The value_counts() method returns a Series object where the index consists of unique values in the 'Job' column, and the values represent the frequency of each unique value.

# .index.tolist(): This part of the code accesses the index of the Series returned by value_counts() and converts it into a list. The index contains unique job titles, and converting it to a list allows us to access its elements more easily.

# [:5]: This part of the code slices the list of unique job titles to select the first five elements only. This effectively gives us the top 5 most frequent job titles in the DataFrame df.

# top_n_jobs = ...: This assigns the list of top 5 most frequent job titles to the variable top_n_jobs.

# print(top_n_jobs): This line prints the list of top 5 most frequent job titles to the console.

# In summary, this code snippet calculates the frequency count of unique job titles in the DataFrame df, extracts the top 5 most frequent job titles, and then prints them out.