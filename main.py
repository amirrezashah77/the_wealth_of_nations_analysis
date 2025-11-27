import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


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
        gdp_df = gdp_df[['REF_AREA_LABEL','REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        gdp_df.columns = ['Country', 'Code', 'Year', 'GDP']

        # Load Life Expectancy Data
        life_df = pd.read_csv(self.life_path)
        life_df = life_df[['REF_AREA_LABEL','REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        life_df.columns = ['Country','Code','Year', 'Life_Expectancy']

        # Merge them together
        data = pd.merge(gdp_df, life_df, on=['Country','Code', 'Year'], how='inner')

        data = data.sort_values(by=['Country', 'Year'])
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
        return avg_gdp, avg_life

    def calculate_correlation(self):
        """Calculates the Pearson Correlation (0 to 1)."""
        corr = self.data[['GDP', 'Life_Expectancy']].corr().iloc[0, 1]
        return corr
    
    def get_r_squared(self):
        """
        Calculates R-Squared (Coefficient of Determination).
        For simple regression, R^2 = Correlation^2.
        """
        corr = self.calculate_correlation()
        return corr ** 2

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
    
    def get_country_data(self, country_name):
        """Filters the data for a specific country (Time Series)."""
        return self.data[self.data['Country'] == country_name].sort_values('Year')

# CLASS 3: CHART BUILDER
class ChartBuilder:
    """
    This class handles the Visualization.
    """
    @staticmethod
    def plot_scatter(data, slope, intercept):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw Scatter
        sns.scatterplot(data=data, x='GDP', y='Life_Expectancy', alpha=0.6, s=80, ax=ax)
        
        # Draw Regression Line
        x_vals = data['GDP']
        y_vals = slope * np.log(x_vals) + intercept
        ax.plot(x_vals, y_vals, color='red', linewidth=2, label='Log-Linear Trend')

        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('GDP per Capita ($) - Log Scale')
        ax.set_ylabel('Life Expectancy (Years)')
        ax.set_title('Cross-Sectional Analysis: Wealth vs Health')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        return fig

    @staticmethod
    def plot_time_series(country_data, country_name):
        """
        Draws the evolution of a country over time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
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
    st.set_page_config(page_title="Wealth of Nations", layout="wide")
    st.title("🌍 The Wealth of Nations")
    st.markdown("A statistical analysis of Economic Development and Public Health.")

    # 1. Load Data 
    manager = DataManager()
    @st.cache_data
    def get_data():
        return manager.get_clean_data()
    all_data = get_data()

    # SECTION 1: GLOBAL SNAPSHOT 
    st.header("1. Global Snapshot (Regression Analysis)")
    
    years = sorted(all_data['Year'].unique())
    selected_year = st.sidebar.select_slider("Select Year", options=years, value=2019)
    
    year_data = all_data[all_data['Year'] == selected_year]

    if not year_data.empty:
        analyzer = Analyzer(year_data)
        avg_gdp, avg_life = analyzer.calculate_averages()
        r_squared = analyzer.get_r_squared()
        slope, intercept = analyzer.run_regression()
        avg_gdp_growth = year_data['GDP_Growth'].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg GDP per capita", f"${avg_gdp:,.0f}")
        c2.metric("Avg Life Exp", f"{avg_life:.1f} Yrs")
        c3.metric("Avg GDP per capita Growth", f"{avg_gdp_growth:.1f}%")
        c4.metric("R-Squared (R²)", f"{r_squared:.3f}")
        c5.metric("Slope", f"{slope:.2f}")

        # LEADERBOARD TABLE (Top 30)
        st.subheader(f"Top 30 Economies by GDP ({selected_year})")
        top_30 = year_data.sort_values(by='GDP', ascending=False).head(30)
        st.dataframe(top_30,
            column_config={
                "Country": "Country",
                "Code": "Code",
                "GDP": st.column_config.NumberColumn("GDP per capita ($)", format="$ %d"),
                "Life_Expectancy": st.column_config.NumberColumn("Life Exp (Yrs)", format="%.1f"),
                "GDP_Growth": st.column_config.NumberColumn("GDP per capita Growth (%)",format="%.2f%%"),
            },
            hide_index=True, 
            column_order=['Country', 'Code', 'GDP', 'Life_Expectancy', 'GDP_Growth']
        )
        
        # TABS FOR MAP AND SCATTER PLOT
        st.subheader(f"Visual Analysis for {selected_year}")
        tab1, tab2 = st.tabs([ "📈 Scatter Plot", "🗺️ Global Map"])

        
        with tab1:
            # This is the Scatter Plot Tab
            st.markdown("#### Wealth vs. Health Regression")
            fig1 = ChartBuilder.plot_scatter(year_data, slope, intercept)

            st.pyplot(fig1)
            
            st.info(f"R² of {r_squared:.2f} means {r_squared*100:.1f}% of Life Expectancy variation "
                    f"is explained by GDP.")
            
        with tab2:
            # This is the Map Tab
            st.markdown("#### Global Distribution of Wealth and Health")
            
            # Radio button to choose what data to show on the map
            map_selection = st.radio(
                "Select data to map:", 
                ('GDP', 'Life_Expectancy', 'GDP_Growth'), 
                horizontal=True
            )
            map_title = f"Global {map_selection.replace('_', ' ')} ({selected_year})"
            
            # Call our ChartBuilder function
            fig_map = ChartBuilder.plot_map(year_data, map_selection, map_title)
            
            # Use st.plotly_chart() for Plotly graphs
            st.plotly_chart(fig_map, use_container_width=True)

    # SECTION 2: TIME SERIES ANALYSIS

    st.markdown("---")
    st.header("2. Country Deep Dive (Time Series)")
    
    country_list = sorted(all_data['Country'].unique())
    default_index = country_list.index("China") if "China" in country_list else 0
    selected_country = st.selectbox("Select a Country:", country_list, index=default_index)

    history_analyzer = Analyzer(all_data) 
    country_data = history_analyzer.get_country_data(selected_country)
    
    fig2 = ChartBuilder.plot_time_series(country_data, selected_country)
    st.pyplot(fig2)


if __name__ == "__main__":
    main()