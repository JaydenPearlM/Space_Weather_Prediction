import pandas as pd

import pandas as pd

# Load the file
file_path = "C:/Users/maxwe/Documents/Space_weather_predictor/predictor/cleaned_data.xlsx"

# Use pandas to read the data, specifying the delimiter using a regular expression for spaces/tabs
data = pd.read_excel(file_path)

# Extract the Npackets column
npackets = data['Npackets']

print(npackets)

