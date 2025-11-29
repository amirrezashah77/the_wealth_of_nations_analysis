# The Wealth of Nations: Global Development Analysis

## Overview
This project is an interactive Data Science dashboard built with **Python** and **Streamlit**. It explores the statistical relationship between **Economic Prosperity (GDP)**, **Public Health (Life Expectancy)**, and **Child Mortality** across 60+ years of global history.

The application utilizes **World Bank Open Data** to visualize trends, calculate regressions (Linear and Pooled OLS), and identify patterns in how nations develop over time.

## Key Features

* **Interactive Dashboard:** Filter analysis by year (1960–2023) or drill down into specific countries.
* **Econometric Analysis:**
    * **Simple Regression:** Calculates $R^2$ and slopes for Wealth vs. Health (Log-Linear Model).
    * **Panel Regression:** Runs a Pooled OLS model using `statsmodels` to control for multiple variables across the entire timeline.
    * **Correlation Matrix:** Visualizes relationships between key indicators.
* **Advanced Visualizations:**
    * **Interactive Maps:** Choropleth maps using `Plotly` to view global distributions.
    * **3D Scatter Plots:** Multi-dimensional analysis of Wealth, Health, and Mortality.
    * **Time Series:** Historical line charts tracking country progress.


## Project Structure

The code follows a modular Object-Oriented structure for maintainability:

```text
wealth_of_nations_analysis/
├── data/                   # Contains raw CSV files from World Bank
├── main.py                 # The entry point (Streamlit UI)
├── data_manager.py         # Handles data loading, cleaning, merging, and feature engineering
├── analyzer.py             # Handles statistics, regression models, and correlations
├── chart_builder.py        # Handles generating Seaborn, Matplotlib, and Plotly charts
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation