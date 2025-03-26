import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Create output directory for saving visualizations
output_dir = r'C:\Users\nikun\Desktop\MLPR\Project\AI-Energy-Load\weather_energy_dataset\correlation_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Define file paths and check if they exist
weather_path = r'C:\Users\nikun\Desktop\MLPR\Project\AI-Energy-Load\weather_energy_dataset\weather_energy_15min.csv'
gdelt_path = r'C:\Users\nikun\Desktop\ML_News_Energy_Model\GDELT_gkg\gkg_datasets\processed_gkg_15min_intervals.csv'

print("Checking if files exist...")
print(f"Weather file exists: {os.path.exists(weather_path)}")
print(f"GDELT file exists: {os.path.exists(gdelt_path)}")

# Try to find the GDELT file in the project directory if the original path doesn't work
project_dir = r'C:\Users\nikun\Desktop\MLPR\Project'
possible_gdelt_paths = []

# Search for the GDELT file in the project directory
print("\nSearching for 'processed_gkg_15min_intervals.csv' in the project directory...")
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file == "processed_gkg_15min_intervals.csv":
            possible_gdelt_path = os.path.join(root, file)
            possible_gdelt_paths.append(possible_gdelt_path)
            print(f"Found at: {possible_gdelt_path}")

# If found, use the first match, otherwise keep the original path for now
if possible_gdelt_paths:
    gdelt_path = possible_gdelt_paths[0]
    print(f"\nUsing found GDELT file at: {gdelt_path}")
else:
    print("\nCould not find the GDELT file in the project directory.")
    print("Please ensure the file exists or update the path. Proceeding with original path.")

# Load the weather data
try:
    weather_df = pd.read_csv(weather_path)
    print("\n--- WEATHER DATASET INFO ---")
    print("Shape:", weather_df.shape)
    print("\nColumns:", weather_df.columns.tolist())
except FileNotFoundError:
    print(f"ERROR: Cannot find weather file at {weather_path}")
    exit(1)

# Try loading the GDELT data
try:
    gdelt_df = pd.read_csv(gdelt_path)
    print("\n--- GDELT DATASET INFO ---")
    print("Shape:", gdelt_df.shape)
    print("\nColumns:", gdelt_df.columns.tolist())
except FileNotFoundError:
    print(f"\nERROR: Cannot find GDELT file at {gdelt_path}")
    print("You need to update the path to the GDELT file. Please:")
    print("1. Check if the file exists in your system")
    print("2. Update the 'gdelt_path' variable with the correct path")
    print("3. Re-run the script")
    exit(1)

# If we get here, both files were successfully loaded
# Continue with the rest of your analysis...

# Display basic information about each dataset
print("\nFirst 5 rows:")
print(weather_df.head())
print("\nData types:")
print(weather_df.dtypes)
print("\nSummary statistics:")
print(weather_df.describe())

print("\n\n--- GDELT DATASET INFO ---")
print("First 5 rows:")
print(gdelt_df.head())
print("\nData types:")
print(gdelt_df.dtypes)

# Convert datetime columns to pandas datetime format
weather_df['datetime'] = pd.to_datetime(weather_df['timestamp'])
gdelt_df['datetime'] = pd.to_datetime(gdelt_df['time_window'])

# Display date ranges for both datasets
print("\n--- DATE RANGE INFO ---")
print("Weather data date range:", weather_df['datetime'].min(), "to", weather_df['datetime'].max())
print("GDELT data date range:", gdelt_df['datetime'].min(), "to", gdelt_df['datetime'].max())

# Merge datasets on datetime
merged_df = pd.merge(weather_df, gdelt_df, left_on='datetime', right_on='datetime', how='inner')

# Check the shape of the merged dataset
print("\n--- MERGED DATASET INFO ---")
print("Shape:", merged_df.shape)
print("Date range:", merged_df['datetime'].min(), "to", merged_df['datetime'].max())
print("Columns:", merged_df.columns.tolist())

# Show missing values per column
print("\nMissing values per column:")
print(merged_df.isna().sum())

# Fill missing values if necessary
merged_df = merged_df.fillna(0)

# Save the merged dataset
merged_df.to_csv(os.path.join(output_dir, 'merged_weather_gdelt_data.csv'), index=False)

# Set target variable
target = 'Power demand_sum'
print(f"\nTarget variable: {target}")
print(f"Target variable statistics:\n{merged_df[target].describe()}")

# Select numerical columns for correlation analysis
numerical_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
correlation_matrix = merged_df[numerical_cols].corr()

# Save correlation with target variable
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)
correlation_with_target.to_csv(os.path.join(output_dir, 'correlation_with_energy_load.csv'))

print("\nTop 10 features correlated with Power demand:")
print(correlation_with_target.head(11))  # 11 because it includes itself

# Create and save correlation heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_correlation_heatmap.png'), dpi=300)

# Create a more focused heatmap with top correlated features
top_features = correlation_with_target.abs().nlargest(15).index
top_correlation = merged_df[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(top_correlation, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Top 15 Features Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_features_correlation_heatmap.png'), dpi=300)

# Create time series plots for energy load and top correlated features
plt.figure(figsize=(16, 8))
merged_df.set_index('datetime')[target].plot()
plt.title('Energy Load Over Time')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_load_time_series.png'), dpi=300)

# Plot top weather and GDELT features over time
weather_features = [col for col in top_features if col in weather_df.columns]
gdelt_features = [col for col in top_features if col in gdelt_df.columns]

if weather_features:
    plt.figure(figsize=(16, 8))
    for feature in weather_features[:3]:  # Limit to top 3 for clarity
        merged_df.set_index('datetime')[feature].plot(label=feature)
    plt.legend()
    plt.title('Top Weather Features Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_weather_features_time_series.png'), dpi=300)

if gdelt_features:
    plt.figure(figsize=(16, 8))
    for feature in gdelt_features[:3]:  # Limit to top 3 for clarity
        merged_df.set_index('datetime')[feature].plot(label=feature)
    plt.legend()
    plt.title('Top GDELT Features Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_gdelt_features_time_series.png'), dpi=300)

# Feature selection recommendation
def select_features(correlation_with_target, threshold=0.2):
    important_features = correlation_with_target[
        (correlation_with_target.abs() > threshold) & 
        (correlation_with_target.index != target)
    ].index.tolist()
    return important_features

# Select features with moderate to strong correlation
selected_features = select_features(correlation_with_target)
print("\nSelected features for prediction model:")
print(selected_features)

# Save selected features to file
pd.Series(selected_features).to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False, header=['feature_name'])

print(f"\nAnalysis complete! Results saved to {output_dir} directory.")


