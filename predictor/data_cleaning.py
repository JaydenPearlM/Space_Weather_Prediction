import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_preprocessed_data(file_path):
    try:
        # Load the cleaned Excel file directly
        df = pd.read_excel(file_path)

        # Check for missing values and fill them (if necessary)
        if df.isnull().sum().any():
            print("Missing values detected, filling with forward fill.")
            df.fillna(method='ffill', inplace=True)

        # Normalize features (only numeric columns)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_columns.empty:
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            print("Data after cleaning and normalization:\n", df.head())
        else:
            print("No numeric columns to normalize.")

    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return df

if __name__ == "__main__":
    # Use a single file path variable for consistency
    file_path = 'C:/Users/maxwe/Documents/Space_weather_predictor/datasets/Solar_Flare/cleaned_data/cleaned_Data.xlsx'
    cleaned_data = load_preprocessed_data(file_path)

    if cleaned_data is not None:
        # Split the data into training and testing sets
        X = cleaned_data.drop(columns=['Filename'])  # Replace 'Filename' with your target column if needed
        y = cleaned_data['Status_flag']  # Replace with your actual target column

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training and testing sets created successfully.")
