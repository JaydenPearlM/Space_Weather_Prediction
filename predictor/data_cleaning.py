import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # Step 1: Load data (example with txt file, assuming space or tab separated)
    try:
        df = pd.read_table(file_path, delimiter='\t', header=None, skiprows=3)
        print(f"Data loaded successfully from {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Step 2: Inspect the first few rows of the data
    print("First few rows of the dataset:")
    print(df.head(10))

    # 3.1 Handle missing values (drop or impute)
    df_cleaned = df.dropna(how='all')
    print("Missing values have been cleaned and removed.")
    
    df_cleaned.drop_duplicates(inplace=True) 
    print("Data after removing duplicates:\n", df_cleaned.head())      
    
    return df_cleaned 


def save_cleaned_data(df, file_path):
    try:
        if df.empty:
            print("DataFrame is empty, nothing to save.")
        else:
            df.to_csv(file_path, index=False)
            print(f"Cleaned data saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    cleaned_data = load_and_clean_data('C:/Users/maxwe/Documents/Space_weather_predictor/datasets/Solar_Flare/hsi_filedb_200510.txt')
    if cleaned_data is not None:
        print("Cleaned data:")
        print(cleaned_data.head(10))        
        print("Missing values before handling:\n", cleaned_data.isnull().sum())
        print("'Missing values after handling:\n", cleaned_data.isnull().sum())
        
        
        save_cleaned_data(cleaned_data, 'C:/Users/maxwe/Documents/Space_weather_predictor/cleaned_solar_flare_data.csv')
    else:
        print("Data loading failed. Check the error above.")
        
        
        
        
        
