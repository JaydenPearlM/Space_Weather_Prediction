import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(df):
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix on numeric columns
    correlation = numeric_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    # Load your cleaned data (adjust the path)
    file_path = r'C:\Users\maxwe\Documents\Space_weather_predictor\datasets\Solar_Flare\Csv_cleaned_Data\preprocessed_data.xlsx'
    df = pd.read_excel(file_path)

    # Call the heatmap function
    plot_heatmap(df)
