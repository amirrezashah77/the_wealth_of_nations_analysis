import streamlit as st

from data_manager import DataManager
from analyzer import Analyzer
from chart_builder import ChartBuilder


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
            
            map_options = {
                "GDP per capita": "GDP",
                "Life Expectancy": "Life_Expectancy",
                "Child Mortality": "Mortality_Rate",
                "GDP per capita Growth": "GDP_Growth"
            }
            
            selected_label = st.radio("Select Variable to Map:", 
                                      options=list(map_options.keys()), 
                                      horizontal=True)
            
            column_name = map_options[selected_label]
            
            fig_map = ChartBuilder.plot_map(year_data, column_name, f"Global {selected_label} ({selected_year})")
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
        with st.expander(f"Click to see Top 30 Economies by GDP per capita in {selected_year}"):
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
        
        # ANALYTICAL MARKDOWN for panel regression
        st.markdown(f"""
        ### 📉 Regression Analysis
        This model controls for both variables simultaneously using data from all years.
        
        1.  **Effect of Wealth:** The coefficient for `log(GDP)` is **{results.params['np.log(GDP)']:.2f}**. This isolates the impact of wealth alone: holding mortality constant, a 1% increase in GDP leads to a statistically significant increase in life expectancy.
        2.  **Effect of Mortality:** The coefficient for `Mortality_Rate` is **{results.params['Mortality_Rate']:.2f}**. This is the independent negative impact of child deaths on the national average age, separate from the country's income level.
        """)

    # SECTION 3: TIME SERIES
    st.markdown("---")
    st.header("3. time Series Analysis: Country Evolution Over Time")
    
    country_list = sorted(all_data['Country'].unique())
    default_country = "Italy" if "Italy" in country_list else country_list[0]
    selected_country = st.selectbox("Select a Country:", country_list, index=country_list.index(default_country))

    history_analyzer = Analyzer(all_data)
    country_data = history_analyzer.get_country_data(selected_country)
    
    fig_ts = ChartBuilder.plot_time_series(country_data, selected_country)
    st.pyplot(fig_ts)
    
    # ANALYTICAL MARKDOWN for time series
    st.markdown("""
    ### ⏳ Quick Takeaways
    * **Development Order:** Usually, the red line (Mortality) drops *before* the blue line (Life Exp) shoots up.
    * **The COVID Shock:** Look at the very end of the charts (2020-2021). You will often see a sharp dip in Life Expectancy and GDP per capita clear evidence of the pandemic's impact.
    """)

if __name__ == "__main__":
    main()