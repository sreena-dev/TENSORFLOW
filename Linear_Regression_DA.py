import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import pandas as pd

df=keras.datasets.boston_housing

(x_train,y_train),(x_test,y_test)=df.load_data()

feature_names = [
    'CRIM',    # per capita crime rate by town
    'ZN',      # proportion of residential land zoned for lots over 25,000 sq.ft.
    'INDUS',   # proportion of non-retail business acres per town
    'CHAS',    # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    'NOX',     # nitric oxides concentration (parts per 10 million)
    'RM',      # average number of rooms per dwelling
    'AGE',     # proportion of owner-occupied units built prior to 1940
    'DIS',     # weighted distances to five Boston employment centres
    'RAD',     # index of accessibility to radial highways
    'TAX',     # full-value property-tax rate per $10,000
    'PTRATIO', # pupil-teacher ratio by town
    'B',       # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    'LSTAT'    # % lower status of the population
]

df = pd.DataFrame(x_train, columns=feature_names)
y_train_series = pd.Series(y_train, name='MEDV')
train_data_combined = pd.concat([df, y_train_series], axis=1)

print("--- Combined Training Data (first 5 rows) ---")
print(train_data_combined.head())

print("\n--- Basic Information about the DataFrame (Data Types, Non-Null Counts) ---")
train_data_combined.info()

print("\n--- Descriptive Statistics for Numerical Columns ---")
# This gives mean, std, min, max, and quartiles for each numerical column
print(train_data_combined.describe())

print("\n--- Checking for Missing Values (Sum of NaNs per column) ---")
print(train_data_combined.isnull().sum())

print("\n--- Selecting Specific Columns ---")
# Select a single column
print("\n'RM' (Average rooms per dwelling) column (first 5 values):")
print(train_data_combined['RM'].head())

# Select multiple columns
print("\n'RM' and 'LSTAT' (Lower status population) columns (first 5 rows):")
print(train_data_combined[['RM', 'LSTAT']].head())

print("\n--- Filtering Rows Based on Conditions ---")
# Filter houses with more than 7 rooms
print("\nHouses with more than 7 rooms (first 5 rows):")
print(train_data_combined[train_data_combined['RM'] > 7].head())

