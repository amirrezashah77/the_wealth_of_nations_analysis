import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataManager:
    """
    This class handles loading and cleaning and merging the data.
    """
    def __init__(self):

        self.gdp_path = "data/gdp.csv"
        self.life_path = "data/life_expectancy.csv"

    def get_clean_data(self):
        # Load GDP Data
        gdp_df = pd.read_csv(self.gdp_path)
        
        # Keep only the columns we need: Country Name, Year, and the Value
        gdp_df = gdp_df[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']]
        gdp_df.columns = ['Country', 'Year', 'GDP']

        # Load Life Expectancy Data
        life_df = pd.read_csv(self.life_path)
        life_df = life_df[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']]
        life_df.columns = ['Country', 'Year', 'Life_Expectancy']

        # Merge them together
        data = pd.merge(gdp_df, life_df, on=['Country', 'Year'], how='inner')
        
        return data
    


class Analyzer:
    """
    This class handles the statistics like correlation and regression.
    """
    def __init__(self, data):
        self.data = data

    def calculate_averages(self):
        avg_gdp = self.data['GDP'].mean()
        avg_life = self.data['Life_Expectancy'].mean()
        return avg_gdp, avg_life

    def calculate_correlation(self):
        """Calculates the Pearson Correlation (0 to 1)."""
        corr = self.data[['GDP', 'Life_Expectancy']].corr().iloc[0, 1]
        return corr

    def run_regression(self):
        """
        Uses NumPy to calculate a Linear Regression line.
        Formula: y = mx + b (Slope and Intercept)
        """
        # X axis = GDP, Y axis = Life Expectancy
        x = self.data['GDP']
        y = self.data['Life_Expectancy']
        
        # Perform log transformation on GDP for better linearity
        slope, intercept = np.polyfit(np.log(x), y, 1) 
        
        return slope, intercept







