import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


class Analyzer:
    """
    This class handles the statistics like correlation, regression, and filtering.
    """
    def __init__(self, data):
        self.data = data

    def calculate_averages(self):
        avg_gdp = self.data['GDP'].mean()
        avg_life = self.data['Life_Expectancy'].mean()
        avg_mortality = self.data['Mortality_Rate'].mean()
        return avg_gdp, avg_life, avg_mortality

    def get_correlation_matrix(self):

        """
        Calculates the correlation between GDP, Life Exp, and Mortality.
        """
        cols = ['GDP', 'Life_Expectancy', 'Mortality_Rate']
        
        # Calculate matrix
        matrix = self.data[cols].corr()
        
        # --- NEW LINE: Rename 'GDP' to 'GDP per capita' for display ---
        return matrix.rename(index={'GDP': 'GDP per capita'}, columns={'GDP': 'GDP per capita'})
    
    def calculate_regression_stats(self, x_col, y_col, use_log=False):
        """
        Calculates Slope, Intercept, and R-Squared for any two columns.
        """
        # 1. Prepare data (drop NaNs for these 2 specific columns)
        # We need to filter out zeros if using log
        subset = self.data[[x_col, y_col]].dropna()
        
        if use_log:
            subset = subset[subset[x_col] > 0]
            x = np.log(subset[x_col])
        else:
            x = subset[x_col]
            
        y = subset[y_col]
        
        # 2. Correlation & R-Squared
        corr = x.corr(y)
        r_squared = corr ** 2
        
        # 3. Slope & Intercept
        slope, intercept = np.polyfit(x, y, 1)
        
        return r_squared, slope, intercept

    def run_regression(self):
        """
        Simple 1-variable regression for the Scatter Plot line.
        """
        x = self.data['GDP']
        y = self.data['Life_Expectancy']
        
        # Check to avoid log(0) errors
        valid_data = self.data[self.data['GDP'] > 0]
        
        slope, intercept = np.polyfit(np.log(valid_data['GDP']), valid_data['Life_Expectancy'], 1) 
        return slope, intercept
    
        
    
    def run_panel_regression(self):
        """
        Runs a Pooled OLS Regression on the entire Panel Dataset.
        Equation: Life_Expectancy = B0 + B1*log(GDP) + B2*Mortality_Rate
        """
        # We use 'np.log()' inside the formula because GDP is exponential
        # This formula tells statsmodels exactly what math to do
        model = smf.ols("Life_Expectancy ~ np.log(GDP) + Mortality_Rate", data=self.data)
        results = model.fit()
        return results
    
    def get_country_data(self, country_name):
        """Filters the data for a specific country (Time Series)."""
        return self.data[self.data['Country'] == country_name].sort_values('Year')
