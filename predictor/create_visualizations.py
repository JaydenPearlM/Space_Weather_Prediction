import pandas as pd

# Load the data from your Excel sheet
df = pd.read_excel(r"C:/Users/maxwe/Documents/Space_weather_predictor/predictor/cleaned_data.xlsx") # Change to your Excel file path

# Check the first few rows
print(df.head(10))
