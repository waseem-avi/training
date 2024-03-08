import pandas as pd

# Load the Walmart Stock CSV file and have pandas infer the data types
walmart_stock = pd.read_csv('Mar 8/walmart_stock.csv')

# 2. Column names
column_names = walmart_stock.columns.tolist()
print("Column names:", column_names)

# 3. Schema
print("Schema:")
print(walmart_stock.dtypes)

# 4. Print out the first 5 rows
print("First 5 rows:")
print(walmart_stock.head())

# 5. Describe the DataFrame
print("Description:")
print(walmart_stock.describe())



# 7. Calculate HV Ratio
walmart_stock['HV Ratio'] = walmart_stock['High'] / walmart_stock['Volume']
print("HV Ratio:")
print(walmart_stock['HV Ratio'])

# 6. Format describe() output to show up to two decimal places
pd.options.display.float_format = '{:.2f}'.format
print("Formatted Description:")
print(walmart_stock.describe())

# 8. Day with the Peak High in Price
peak_high_day = walmart_stock.loc[walmart_stock['High'].idxmax()]['Date']
print("Peak High Day:", peak_high_day)

# 9. Max and Min of the Volume column
max_volume = walmart_stock['Volume'].max()
min_volume = walmart_stock['Volume'].min()
print("Max Volume:", max_volume)
print("Min Volume:", min_volume)

# 10. Number of days the Close was lower than 60 dollars
close_lower_than_60 = walmart_stock[walmart_stock['Close'] < 60]['Date'].count()
print("Days Close Lower Than $60:", close_lower_than_60)

# 11. Percentage of time the High was greater than 80 dollars
high_greater_than_80 = (walmart_stock[walmart_stock['High'] > 80]['Date'].count() / len(walmart_stock)) * 100
print("Percentage of Time High > $80:", high_greater_than_80)

# 12. Pearson correlation between High and Volume
pearson_corr = walmart_stock['High'].corr(walmart_stock['Volume'])
print("Pearson correlation between High and Volume:", pearson_corr)

# 13. Max High per year
walmart_stock['Year'] = pd.to_datetime(walmart_stock['Date']).dt.year
max_high_per_year = walmart_stock.groupby('Year')['High'].max()
print("Max High per Year:")
print(max_high_per_year)

# 14. Average Close for each Calendar Month
walmart_stock['Month'] = pd.to_datetime(walmart_stock['Date']).dt.month
average_close_per_month = walmart_stock.groupby('Month')['Close'].mean()
print("Average Close for each Calendar Month:")
print(average_close_per_month)