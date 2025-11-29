import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np


class ChartBuilder:
    """
    This class handles the Visualization.
    """

    @staticmethod
    def plot_correlation_heatmap(corr_matrix):
        """
        Draws a Heatmap to visualize the correlation matrix.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig
    
    @staticmethod
    def plot_scatter(data, x_col, y_col, slope, intercept, r2, title, xlabel, use_log=False):
        """
        Generic scatter plotter that can handle both GDP (Log) and Mortality (Linear).
        Displays Slope and Intercept in the title.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 1. Draw Scatter
        sns.scatterplot(data=data, x=x_col, y=y_col, alpha=0.6, s=80, ax=ax)
        
        # 2. Draw Regression Line
        # Remove NaNs/Zeros for plotting line
        if use_log:
            clean_data = data[data[x_col] > 0]
            x_vals = clean_data[x_col]
            y_vals = slope * np.log(x_vals) + intercept
            ax.set_xscale('log') # Log Scale for GDP
            line_color = 'red'
        else:
            x_vals = data[x_col]
            y_vals = slope * x_vals + intercept
            line_color = 'orange' # Different color for Mortality

        ax.plot(x_vals, y_vals, color=line_color, linewidth=3, label='Trend Line')

        # 3. Dynamic Title with Stats
        # We put the math in the subtitle
        full_title = f"{title}\n(Slope: {slope:.2f} | Intercept: {intercept:.2f} | R²: {r2:.2f})"
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Life Expectancy (Years)")
        ax.set_title(full_title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        return fig
    
    @staticmethod
    def plot_3d_scatter(data):
        """
        Uses Plotly to draw a 3D chart since we have 3 variables now.
        """
        fig = px.scatter_3d(
            data,
            x='GDP',
            y='Mortality_Rate',
            z='Life_Expectancy',
            color='Country',
            log_x=True, # Log scale for GDP per capita
            title="3D Analysis: Wealth, Health & Mortality",
            height=600,
            labels={'GDP': 'GDP per capita'}
        )
        return fig

    @staticmethod
    def plot_time_series(country_data, country_name):
        """
        Draws the evolution of a country over time.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: GDP
        sns.lineplot(data=country_data, x='Year', y='GDP', ax=ax1, color='green', linewidth=2)
        ax1.set_title(f"{country_name}: Economic Growth")
        ax1.set_ylabel("GDP per capita ($)")
        ax1.grid(True)
        
        # Plot 2: Life Expectancy
        sns.lineplot(data=country_data, x='Year', y='Life_Expectancy', ax=ax2, color='blue', linewidth=2)
        ax2.set_title(f"{country_name}: Health Improvement")
        ax2.set_ylabel("Life Expectancy (Years)")
        ax2.grid(True)

        # Plot 3: Mortality Rate
        sns.lineplot(data=country_data, x='Year', y='Mortality_Rate', ax=ax3, color='red', linewidth=2)
        ax3.set_title(f"{country_name}: Child Mortality Rate")
        ax3.set_ylabel("Mortality Rate (per 1,000 live births)")        
        ax3.grid(True)     
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_map(data, column_to_plot, title):
        """
        Creates an interactive Plotly World Map.
        """
        import plotly.express as px
        
        fig = px.choropleth(
            data,
            locations="Code",             
            color=column_to_plot,       
            hover_name="Country",     
            color_continuous_scale=px.colors.sequential.Plasma, 
            title=title,    
            labels={'GDP': 'GDP per capita', 'Mortality_Rate': 'Child Mortality'}
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    