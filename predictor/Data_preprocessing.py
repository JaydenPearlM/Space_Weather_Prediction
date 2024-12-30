import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_preprocessed_data(file_path):
    # Define column names based on the dataset's structure
    columns = ['Filename', 'Orb_st', 'Orb_end', 'Start_time', 'End_time', 'Status_flag', 'Npackets', 'Drift_start', 'Drift_end', 'Data_source']
    data = []
    
    try:
        # Read the file and skip the first 4 lines (metadata)
        with open(file_path, 'r') as file:
            lines = file.readlines()[4:]  # Skip the first 4 lines which are metadata
            
            print(f"Total lines read (after skipping metadata): {len(lines)}")
            
            for line in lines:
                # Split line into parts based on spaces and strip extra spaces
                parts = line.strip().split()
                
                # Clean up the 'Filename' column (remove extra quotes)
                if parts[0].startswith('"') and parts[0].endswith('"'):
                    parts[0] = parts[0][1:-1]  # Remove the surrounding quotes
                
                # Debug: print the parts to ensure data is split correctly
                print("Parts of the line:", parts)
                
                if len(parts) >= 10:  # Ensure there are at least 10 parts
                    data.append(parts[:10])  # Only take the first 10 columns
                    
        if data:  # Check if data has been successfully loaded
            # Create DataFrame with the specified column names
            df = pd.DataFrame(data, columns=columns)
            
            # Verify the column names after loading
            print("Verified column names in data:", df.columns)
            
            # Debug: Print the first few rows of data to verify
            print("First few rows of the dataset:")
            print(df.head())

            # Check data types of columns
            print("Data types of columns:\n", df.dtypes)

            # Convert numeric columns to proper numeric types, coerce errors to NaN
            numeric_columns = ['Orb_st', 'Orb_end', 'Npackets', 'Drift_start', 'Drift_end']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            print("Data types after conversion:\n", df.dtypes)

            # Check for missing values and fill them (if necessary)
            if df.isnull().sum().any():
                print("Missing values detected, filling with forward fill.")
                df.fillna(method='ffill', inplace=True)

            # Normalize features (only numeric columns)
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if numeric_columns.any():
                scaler = MinMaxScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                print("Data after cleaning and normalization:\n", df.head())
                df.to_csv("cleaned_data.csv", index=False)
                print("cleaning_data.csv successfully saved")
            else:
                print("No numeric columns to normalize.")
        else:
            print("No data found in the file after skipping metadata.")
            return None
        
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return df

if __name__ == "__main__":
    # Use a single file path variable for consistency
    file_path = 'C:/Users/maxwe/Documents/Space_weather_predictor/predictor/cleaned_solar_flare_data.csv'
    cleaned_data = load_preprocessed_data(file_path)

    if cleaned_data is not None:
        # Split the data into training and testing sets
        X = cleaned_data.drop(columns=['Filename'])  # Replace 'Filename' with your target column if needed
        y = cleaned_data['End_time']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X = cleaned_data[['Start_time', 'NpEnd_time']]  # Feature columns (adjust based on your dataset)
        y = cleaned_data['Status_flag']  # Target column (adjust based on your goal)

# Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
