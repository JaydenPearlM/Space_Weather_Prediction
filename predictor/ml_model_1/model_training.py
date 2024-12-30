import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # For classification tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # For classification evaluation
from sklearn.preprocessing import StandardScaler  # Optional for scaling
import joblib

# Load the cleaned data (CSV)
data_path = 'C:/Users/maxwe/Documents/Space_weather_predictor/predictor/cleaned_data.xlsx'
data = pd.read_excel(data_path)

# Print the column names to verify the correct columns
print("Columns in the dataset:")
print(data.columns)

# Check the first few rows of the dataset to verify the data
print("First few rows of the dataset:")
print(data.head(10))

# Ensure that 'Start_time' is available and correctly formatted
if 'Start_time' in data.columns:
    # Convert 'Start_time' to datetime (if it's in string format) and then to int64 (seconds)
    data['Start_time'] = pd.to_datetime(data['Start_time'], errors='coerce').astype('int64') // 10**9
    print("Converted Start_time to datetime.")
else:
    print("'Start_time' column is missing or misnamed.")
    exit(1)

# Convert 'End_time' to datetime and then to int64 (seconds), similar to Start_time
if 'End_time' in data.columns:
    data['End_time'] = pd.to_datetime(data['End_time'], errors='coerce').astype('int64') // 10**9
    print("Converted End_time to datetime.")
else:
    print("'End_time' column is missing or misnamed.")
    exit(1)

# Drop rows with missing values in critical columns
data = data.dropna(subset=['Start_time', 'End_time', 'Status_flag'])

# Check if data is empty after cleaning
if data.empty:
    print("No valid data left after cleaning. Please check the dataset.")
    exit(1)

# Define the feature columns (Start_time and End_time) and the target column (Status_flag)
X = data[['Start_time', 'End_time']]  # Feature columns
y = data['Status_flag']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature columns using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model (e.g., Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
