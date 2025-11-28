import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.formula.api as smf

class DataManager:
    """
    This class handles loading and cleaning and merging the data.
    """
    def __init__(self):

        self.gdp_path = "data/gdp.csv"
        self.life_path = "data/life_expectancy.csv"
        self.mortality_path = "data/child_mortality.csv"

    def get_clean_data(self):
        # Load GDP Data
        gdp_df = pd.read_csv(self.gdp_path)
        gdp_df = gdp_df[['REF_AREA_LABEL','REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        gdp_df.columns = ['Country', 'Code', 'Year', 'GDP']

        # Load Life Expectancy Data
        life_df = pd.read_csv(self.life_path)
        life_df = life_df[['REF_AREA_LABEL','REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        life_df.columns = ['Country','Code','Year', 'Life_Expectancy']

        # LOAD & CLEAN MORTALITY
        mortality_df = pd.read_csv(self.mortality_path)
        mortality_df = mortality_df[['REF_AREA_LABEL', 'REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        mortality_df.columns = ['Country', 'Code', 'Year', 'Mortality_Rate']

        # Merge them together
        merged_data = pd.merge(gdp_df, life_df, on=['Country','Code', 'Year'], how='inner')
        # Merge Mortality data
        merged_data = pd.merge(merged_data, mortality_df, on=['Country', 'Code', 'Year'], how='inner')

        data = merged_data.sort_values(by=['Country', 'Year'])
        # Calculate GDP Growth Rate
        data['GDP_Last_Year'] = data.groupby('Country')['GDP'].shift(1)
        data['GDP_Growth'] = (data['GDP'] - data['GDP_Last_Year']) / data['GDP_Last_Year']* 100
        
        return data
    


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
        # Select numeric columns only
        cols = ['GDP', 'Life_Expectancy', 'Mortality_Rate']
        return self.data[cols].corr()
    
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

# CLASS 3: CHART BUILDER
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
            height=600
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
            title=title
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
#  THE MAIN APPLICATION     
def main():
    st.set_page_config(page_title="Wealth & Health Analysis", layout="wide")
    st.title("🌍 Global Development: Wealth, Health & Mortality")
    st.markdown("A statistical analysis of how **Economic Prosperity** and **Child Mortality** impact **Life Expectancy**.")

    # 1. Load Data
    manager = DataManager()
    @st.cache_data
    def get_data():
        return manager.get_clean_data()
    
    all_data = get_data()

    #SECTION 1: cross-Sectional ANALYSIS
    st.header("1.cross-Sectional Analysis: Yearly Snapshot")
    
    years = sorted(all_data['Year'].unique())
    selected_year = st.sidebar.select_slider("Select Year", options=years, value=2019)
    
    year_data = all_data[all_data['Year'] == selected_year]

    if not year_data.empty:
        analyzer = Analyzer(year_data)
        
        # Calculate Stats
        avg_gdp, avg_life, avg_mort = analyzer.calculate_averages()
        r2_gdp, slope_gdp, intercept_gdp = analyzer.calculate_regression_stats('GDP', 'Life_Expectancy', use_log=True)
        r2_mort, slope_mort, intercept_mort = analyzer.calculate_regression_stats('Mortality_Rate', 'Life_Expectancy', use_log=False)
        corr_matrix = analyzer.get_correlation_matrix()

        # Display Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg GDP per capita", f"${avg_gdp:,.0f}")
        c2.metric("Avg Life Exp", f"{avg_life:.1f} Yrs")
        c3.metric("Avg Mortality", f"{avg_mort:.1f}")
        c4.metric("R² (log(GDP per capita) vs Life)", f"{r2_gdp:.2f}")
        c5.metric("R² (Mortality vs Life)", f"{r2_mort:.2f}")

        # VISUALIZATION TABS 
        st.subheader(f"Visual Analysis for {selected_year}")
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Global Map", "📈 Scatter Plots", "🔥 Correlation Matrix", "🧊 3D Analysis"])

        with tab1:
            st.markdown("#### Interactive Global Maps")
            map_var = st.radio("Select Variable to Map:", 
                               ['GDP', 'Life_Expectancy', 'Mortality_Rate', 'GDP_Growth'], 
                               horizontal=True)
            display_title = "GDP per capita" if map_var == 'GDP' else map_var
            fig_map = ChartBuilder.plot_map(year_data, map_var, f"Global {display_title} ({selected_year})")
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            st.markdown("#### Regression Analysis")
            graph_choice = st.radio("Select Regression Model:", 
                                    ["Wealth vs Health (Log-Linear)", "Mortality vs Health (Linear)"], 
                                    horizontal=True)
            
            if graph_choice == "Wealth vs Health (Log-Linear)":
                fig_scatter = ChartBuilder.plot_scatter(
                    year_data, 'GDP', 'Life_Expectancy', slope_gdp, intercept_gdp, r2_gdp,
                    "Wealth vs Health", "GDP per capita ($) - Log Scale", use_log=True
                )
                st.pyplot(fig_scatter)
                
                # ANALYTICAL MARKDOWN  for GDP vs Life Expectancy
                st.markdown(f"""
                ### 📊 Statistical Interpretation
                * **Model Fit ($R^2 = {r2_gdp:.2f}$):** GDP explains approximately **{r2_gdp*100:.1f}%** of the variation in Life Expectancy across countries.
                * **Slope Coefficient ({slope_gdp:.2f}):** The positive slope confirms a direct relationship. However, since the X-axis is logarithmic, the curve flattens at the top. This indicates that while wealth improves health rapidly for developing nations, developed nations see smaller marginal gains from additional wealth.
                """)

            else:
                fig_scatter = ChartBuilder.plot_scatter(
                    year_data, 'Mortality_Rate', 'Life_Expectancy', slope_mort, intercept_mort, r2_mort,
                    "Child Mortality vs Health", "Child Mortality (Deaths per 1,000 births)", use_log=False
                )
                st.pyplot(fig_scatter)
                
                # ANALYTICAL MARKDOWN for Mortality vs Life Expectancy
                st.markdown(f"""
                ### 📊 Statistical Interpretation
                * **Model Fit ($R^2 = {r2_mort:.2f}$):** Child Mortality is a stronger predictor than GDP, explaining **{r2_mort*100:.1f}%** of the variation.
                * **Slope Coefficient ({slope_mort:.2f}):** The negative slope indicates that for every increase of 1 unit in child mortality per 1,000 births, average life expectancy drops by **{abs(slope_mort):.2f} years**.
                """)

        with tab3:
            st.markdown("#### Correlation Heatmap")
            fig_corr = ChartBuilder.plot_correlation_heatmap(corr_matrix)
            st.pyplot(fig_corr)

        with tab4:
            st.markdown("#### 3D Visualization")
            fig_3d = ChartBuilder.plot_3d_scatter(year_data)
            st.plotly_chart(fig_3d, use_container_width=True)

        # Leaderboard Table
        with st.expander(f"Click to see Top 30 Economies in {selected_year}"):
            top_30 = year_data.sort_values(by='GDP', ascending=False).head(30)
            top_30 = top_30.rename(columns={'GDP': 'GDP per capita'})
            st.dataframe(top_30, hide_index=True)

    # SECTION 2: PANEL REGRESSION
    st.markdown("---")
    st.header("2. Advanced Analysis: Panel Regression")
    
    if st.checkbox("Run Panel Regression (Pooled OLS)"):
        panel_analyzer = Analyzer(all_data)
        results = panel_analyzer.run_panel_regression()
        st.write("### Model Results")
        st.code(results.summary().tables[1].as_text(), language='text')
        
        # ANALYTICAL MARKDOWN 3 for panel regression
        st.markdown(f"""
        ### 📉 Regression Analysis
        This model controls for both variables simultaneously using data from all years.
        
        1.  **Effect of Wealth:** The coefficient for `log(GDP)` is **{results.params['np.log(GDP)']:.2f}**. This isolates the impact of wealth alone: holding mortality constant, a 1% increase in GDP leads to a statistically significant increase in life expectancy.
        2.  **Effect of Mortality:** The coefficient for `Mortality_Rate` is **{results.params['Mortality_Rate']:.2f}**. This is the independent negative impact of child deaths on the national average age, separate from the country's income level.
        """)

    # SECTION 3: TIME SERIES
    st.markdown("---")
    st.header("3. Country Deep Dive")
    
    country_list = sorted(all_data['Country'].unique())
    default_country = "China" if "China" in country_list else country_list[0]
    selected_country = st.selectbox("Select a Country:", country_list, index=country_list.index(default_country))

    history_analyzer = Analyzer(all_data)
    country_data = history_analyzer.get_country_data(selected_country)
    
    fig_ts = ChartBuilder.plot_time_series(country_data, selected_country)
    st.pyplot(fig_ts)
    
    # ANALYTICAL MARKDOWN for time series
    st.markdown("""
    ### ⏳ Quick Takeaways
    * **Development Order:** Usually, the red line (Mortality) drops *before* the blue line (Life Exp) shoots up.
    * **The COVID Shock:** Look at the very end of the charts (2020-2021). You will often see a sharp dip in GDP per capita and Life Expectancy—clear evidence of the pandemic's impact.
    """)

if __name__ == "__main__":
    main()