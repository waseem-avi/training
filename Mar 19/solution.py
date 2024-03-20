import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load home price data
home_data = pd.read_csv('Mar 19/homeprices.csv')

# Handling missing values in 'bedrooms' column by filling with median
median_bedrooms = home_data['bedrooms'].median()
home_data['bedrooms'].fillna(median_bedrooms, inplace=True)

# Define features and target for home price prediction
home_features = home_data[['area', 'bedrooms', 'age']]
home_target = home_data['price']

# Train a linear regression model for home price prediction
home_model = LinearRegression()
home_model.fit(home_features, home_target)

# Predicting home prices for given inputs
home_inputs = [[3000, 3, 40], [2500, 4, 5]]
predicted_prices = home_model.predict(home_inputs)

# Print predicted home prices
for i, predicted_price in enumerate(predicted_prices):
    print(f"Predicted price for home {i + 1}: ${predicted_price:.2f}")


# Load hiring data
hiring_data = pd.read_csv('Mar 19/hiring.csv')

# Handling missing values in 'experience' column by filling with zero
hiring_data['experience'].fillna('zero', inplace=True)

# Map experience strings to numerical values
experience_map = {'zero': 0, 'two': 2, 'three': 3, 'five': 5, 'seven': 7, 'ten': 10, 'eleven': 11}
hiring_data['experience'] = hiring_data['experience'].map(experience_map)

# Define features and target for salary prediction
salary_features = hiring_data[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
salary_target = hiring_data['salary($)']

# Handle missing values in salary features
imputer = SimpleImputer(strategy='mean')
salary_features_imputed = imputer.fit_transform(salary_features)

# Train a linear regression model for salary prediction
salary_model = LinearRegression()
salary_model.fit(salary_features_imputed, salary_target)

# Predicting salaries for given candidates
candidate_inputs = [[2, 9, 6], [12, 10, 10]]
predicted_salaries = salary_model.predict(candidate_inputs)

# Print predicted salaries
for i, predicted_salary in enumerate(predicted_salaries):
    print(f"Predicted salary for candidate {i + 1}: ${predicted_salary:.2f}")
