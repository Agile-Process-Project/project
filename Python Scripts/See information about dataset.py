import pandas as pd
import numpy as np

# Load the original Airbnb dataset
df = pd.read_csv("listings_cleaned.csv")

# Display the first 5 rows
print("FIRST FIVE ROWS:")
print(df.head())

# Display the dataset dimensions
print("\nDATASET DIMENSIONS:")
print(df.shape)

# Display column information
print("\nDATASET INFORMATION:")
df.info()

# Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum().sort_values(ascending=False))

# Check for duplicate rows
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Check for duplicate listing IDs
print("\nDUPLICATE LISTING IDs:")
print(df["id"].duplicated().sum())

# Check data types
print("\nDATA TYPES:")
print(df.dtypes)