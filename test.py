from main import DataManager , Analyzer

print("--- STARTING TEST ---")
    
manager = DataManager()
    
print("Attempting to load and merge files...")
df = manager.get_clean_data()
    
if not df.empty:
    print("\nSUCCESS! Data Loaded.")
    print(f"Total Rows Found: {len(df)}")
    print("\nour clean data:")
    print(df.head)  
else:
    print("\nFAILURE: Data frame is empty. Check file paths.")


        
# Filter for a specific year (e.g., 2019)
data_2019 = df[df['Year'] == 2019]
print(f"Data for 2019: {len(data_2019)} countries found.")

if not data_2019.empty:
    analyzer = Analyzer(data_2019)
        
    #Test Averages
    avg_gdp, avg_life = analyzer.calculate_averages()
    print(f"\nAverage GDP: ${avg_gdp:,.2f}")
    print(f"Average Life Expectancy: {avg_life:.1f} Years")
        
    #Test Correlation
    corr = analyzer.calculate_correlation()
    print(f"Correlation: {corr:.4f}")
        
    #Test Regression
    slope, intercept = analyzer.run_regression()
    print(f"Regression Line: Slope={slope:.2f}, Intercept={intercept:.2f}")
    print("(If Slope is positive, it means Money helps Health!)")
        
else:
    print("Error: No data found for 2019.")