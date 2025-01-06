import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_save_preprocessed_data(file_path, output_path):
    try:
        # Load the cleaned Excel file directly
        df = pd.read_excel(file_path)

        # Check for missing values and handle them (if necessary)
        if df.isnull().sum().any():
            print("Missing values detected, filling with forward fill.")
            df.ffill(inplace=True)

        # Normalize features (only numeric columns)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_columns.any():
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            print("Data after cleaning and normalization:\n", df.head())
        else:
            print("No numeric columns to normalize.")

        # Save the preprocessed data to a new Excel file
        df.to_excel(output_path, index=False)  # Save without the index column

        print(f"Preprocessed data saved successfully to {output_path}.")

        # Split the data into features (X) and target (y)
        X = df.drop(columns=['Status_flag'])  # Replace 'Status_flag' with the actual target column if needed
        y = df['Status_flag']  # Replace with your actual target column name if needed

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training and testing sets created successfully.")
        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

if __name__ == "__main__":
    # Use a single file path variable for consistency
    input_file_path = r'C:\Users\maxwe\Documents\Space_weather_predictor\datasets\Solar_Flare\cleaned_Data\cleaned_Data.xlsx'
    output_file_path = r'C:\Users\maxwe\Documents\Space_weather_predictor\datasets\Solar_Flare\Csv_cleaned_Data\preprocessed_data.xlsx'
    
    X_train, X_test, y_train, y_test = load_and_save_preprocessed_data(input_file_path, output_file_path)

    if X_train is not None:
        # Proceed with training and testing data available
        print("Data preprocessing and saving completed.")
