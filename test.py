from data_manager import DataManager
import matplotlib.pyplot as plt
import pandas as pd

# Get the data after merging and feature engineering

manager = DataManager() 

df_check = manager.get_clean_data()

print("\n--- MISSING DATA CHECK ---")
print("Rows before dropping NaNs:", len(df_check))

# This prints the number of missing values for every column
print(df_check.isnull().sum())

