from main import DataManager, Analyzer, ChartBuilder
import matplotlib.pyplot as plt

def run_tests():
    print("--- 🧪 TESTING NEW FEATURES (R2 & TIME SERIES) ---")
    
    # 1. Load Data
    manager = DataManager()
    df = manager.get_clean_data()
    
    if df.empty:
        print("❌ Data Load Failed.")
        return

    # --- TEST A: CROSS-SECTION (SNAPSHOT) ---
    print("\n[A] Testing Regression & R-Squared (Year 2019)...")
    data_2019 = df[df['Year'] == 2019]
    
    analyzer = Analyzer(data_2019)
    slope, intercept = analyzer.run_regression()
    r_sq = analyzer.get_r_squared() # <--- Testing the new function
    
    print(f"   -> Slope: {slope:.2f}")
    print(f"   -> R-Squared: {r_sq:.4f}")
    
    if 0 <= r_sq <= 1:
        print("   ✅ R-Squared is valid (between 0 and 1).")
    else:
        print("   ❌ R-Squared is mathematically impossible.")

    print("   -> Generating Scatter Plot...")
    fig1 = ChartBuilder.plot_scatter(data_2019, slope, intercept)
    fig1.savefig("test_scatter.png")
    print("   ✅ Saved 'test_scatter.png'")

    # --- TEST B: TIME SERIES (HISTORY) ---
    print("\n[B] Testing Time Series Logic...")
    
    # We need to initialize Analyzer with ALL data, not just 2019
    history_analyzer = Analyzer(df)
    
    # Let's try to find a country (e.g., China)
    target_country = "China"
    
    # Check if country exists in data
    if target_country not in df['Country'].values:
        print(f"   ⚠️ {target_country} not found. Picking the first country in the list...")
        target_country = df['Country'].unique()[0]
    
    print(f"   -> Extracting history for: {target_country}")
    country_data = history_analyzer.get_country_data(target_country)
    
    print(f"   -> Found {len(country_data)} years of data.")
    
    print("   -> Generating Time Series Chart...")
    fig2 = ChartBuilder.plot_time_series(country_data, target_country)
    fig2.savefig("test_timeseries.png")
    print("   ✅ Saved 'test_timeseries.png'")

if __name__ == "__main__":
    run_tests()