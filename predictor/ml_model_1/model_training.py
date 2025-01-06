import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the preprocessed data
preprocessed_data_path = r'C:\Users\maxwe\Documents\Space_weather_predictor\datasets\Solar_Flare\Csv_cleaned_Data\preprocessed_data.xlsx'
data = pd.read_excel(preprocessed_data_path)

# Verify the columns and data
print("Columns in the dataset:")
print(data.columns)
print("First few rows of the dataset:")
print(data.head(10))

# Ensure 'Start_time' and 'End_time' are correctly formatted
if 'Start_time' in data.columns:
    data['Start_time'] = pd.to_datetime(data['Start_time'], errors='coerce').astype('int64') // 10**9
    print("Converted 'Start_time' to datetime.")
else:
    raise ValueError("'Start_time' column is missing or misnamed.")

if 'End_time' in data.columns:
    data['End_time'] = pd.to_datetime(data['End_time'], errors='coerce').astype('int64') // 10**9
    print("Converted 'End_time' to datetime.")
else:
    raise ValueError("'End_time' column is missing or misnamed.")

# Drop rows with missing values in critical columns
data = data.dropna(subset=['Start_time', 'End_time', 'Status_flag'])
if data.empty:
    raise ValueError("No valid data left after cleaning. Please check the dataset.")

# Define feature columns and target column
X = data[['Start_time', 'End_time']]
y = data['Status_flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Save the results to an Excel file
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_excel('C:/Users/maxwe/Documents/Space_weather_predictor/datasets/Solar_Flare/actualprediction.xlsx', index=False)

# Evaluate and display model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler for future use
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
