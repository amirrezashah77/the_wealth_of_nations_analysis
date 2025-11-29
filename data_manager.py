import pandas as pd

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