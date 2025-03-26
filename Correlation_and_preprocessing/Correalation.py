import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define project directories
PROJECT_DIR = r'C:\Users\nikun\Desktop\MLPR\AI-Energy-Load-New'
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'OUTPUT_DIR')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'correlation_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define file paths with updated locations
weather_path = os.path.join(PROJECT_DIR, 'weather_energy_dataset', 'weather_energy_15min.csv')
gdelt_path = os.path.join(OUTPUT_DIR, 'aggregated_data', 'aggregated_gkg_15min.csv')

# Create a log file
log_path = os.path.join(RESULTS_DIR, 'correlation_analysis_log.txt')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Starting correlation analysis")
logger.info(f"Weather file exists: {os.path.exists(weather_path)}")
logger.info(f"GDELT file exists: {os.path.exists(gdelt_path)}")

# Load datasets
try:
    weather_df = pd.read_csv(weather_path)
    logger.info(f"Weather data loaded successfully - Shape: {weather_df.shape}")
except FileNotFoundError:
    logger.error(f"Weather file not found at {weather_path}")
    raise

try:
    gdelt_df = pd.read_csv(gdelt_path)
    logger.info(f"GDELT data loaded successfully - Shape: {gdelt_df.shape}")
    logger.info(f"GDELT columns: {', '.join(gdelt_df.columns[:5])}... (total: {len(gdelt_df.columns)})")
except FileNotFoundError:
    logger.error(f"GDELT file not found at {gdelt_path}")
    raise

# Format datetime columns
logger.info("Converting datetime columns...")
weather_df['datetime'] = pd.to_datetime(weather_df['timestamp'])
gdelt_df['datetime'] = pd.to_datetime(gdelt_df['time_bucket'])

# Check date ranges
logger.info(f"Weather data range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
logger.info(f"GDELT data range: {gdelt_df['datetime'].min()} to {gdelt_df['datetime'].max()}")

# Ensure both datasets are sorted by datetime
weather_df = weather_df.sort_values('datetime')
gdelt_df = gdelt_df.sort_values('datetime')

# Merge datasets efficiently
logger.info("Merging datasets on datetime...")
merged_df = pd.merge(weather_df, gdelt_df, on='datetime', how='inner')
logger.info(f"Merged data shape: {merged_df.shape}")
logger.info(f"Merged date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")

# Handle missing values - more sophisticated approach
logger.info("Handling missing values...")
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
missing_counts = merged_df[numeric_cols].isna().sum()
cols_with_missing = missing_counts[missing_counts > 0]

if not cols_with_missing.empty:
    logger.info(f"Columns with missing values: {cols_with_missing.to_dict()}")
    
    # Fill missing values with appropriate methods per column type
    for col in tqdm(cols_with_missing.index, desc="Filling missing values"):
        # For time series data, forward fill is often better than mean/zero
        merged_df[col] = merged_df[col].fillna(method='ffill')
        # If still has NaN (at the beginning), fill with backward fill
        merged_df[col] = merged_df[col].fillna(method='bfill')
        # If still has NaN, use column median (more robust than mean)
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Save the merged dataset
merged_path = os.path.join(RESULTS_DIR, 'merged_weather_gdelt_data.csv')
merged_df.to_csv(merged_path, index=False)
logger.info(f"Merged dataset saved to {merged_path}")

# Define target variable - check if it exists
target_candidates = ['Power demand_sum', 'Power_demand_sum', 'power_demand', 'load']
target = None

for candidate in target_candidates:
    if candidate in merged_df.columns:
        target = candidate
        break

if target is None:
    logger.error("Target variable not found in the dataset! Please check column names.")
    raise ValueError("Target variable not found in dataset")

logger.info(f"Using '{target}' as the target variable")
logger.info(f"Target statistics: min={merged_df[target].min()}, max={merged_df[target].max()}, mean={merged_df[target].mean():.2f}")

# Select numerical columns for correlation analysis - exclude datetime
numerical_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
datetime_cols = [col for col in numerical_cols if 'date' in col.lower() or 'time' in col.lower()]
numerical_cols = [col for col in numerical_cols if col not in datetime_cols]

# Calculate correlation matrix
logger.info("Calculating correlation matrix...")
correlation_matrix = merged_df[numerical_cols].corr()

# Save correlation with target variable
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)
correlation_path = os.path.join(RESULTS_DIR, 'correlation_with_energy_load.csv')
correlation_with_target.to_csv(correlation_path)
logger.info(f"Correlation with target saved to {correlation_path}")

# Log top correlated features
logger.info("Top 10 features correlated with energy demand:")
for feature, corr in correlation_with_target.head(11).items():
    logger.info(f"  {feature}: {corr:.4f}")

# Create and save correlation heatmap
logger.info("Generating correlation heatmaps...")
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Features')
plt.tight_layout()
full_heatmap_path = os.path.join(RESULTS_DIR, 'full_correlation_heatmap.png')
plt.savefig(full_heatmap_path, dpi=300)
plt.close()

# Create a more focused heatmap with top correlated features
top_features = correlation_with_target.abs().nlargest(20).index
top_correlation = merged_df[top_features].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(top_correlation, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Top 20 Features Correlation Heatmap')
plt.tight_layout()
top_heatmap_path = os.path.join(RESULTS_DIR, 'top_features_correlation_heatmap.png')
plt.savefig(top_heatmap_path, dpi=300)
plt.close()

# Group correlated features by category for better understanding
logger.info("Analyzing feature groups...")
feature_categories = {
    'weather': [col for col in top_features if any(x in col.lower() for x in ['temp', 'wind', 'rain', 'humidity', 'pressure'])],
    'news': [col for col in top_features if any(x in col.lower() for x in ['theme', 'tone', 'energy', 'crisis'])],
    'temporal': [col for col in top_features if any(x in col.lower() for x in ['hour', 'day', 'month', 'weekend'])]
}

# Log results by category
for category, features in feature_categories.items():
    if features:
        logger.info(f"Top {category} features:")
        for feature in features:
            logger.info(f"  {feature}: {correlation_with_target[feature]:.4f}")

# Create time series plots with the most important features
logger.info("Generating time series visualizations...")
sample_size = min(10000, len(merged_df))  # Use a reasonable sample size
sample_df = merged_df.sort_values('datetime').iloc[-sample_size:]  # Most recent data

plt.figure(figsize=(16, 8))
plt.plot(sample_df['datetime'], sample_df[target], label=target)
plt.title('Energy Load Time Series')
plt.xlabel('Date')
plt.ylabel('Power Demand')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
energy_ts_path = os.path.join(RESULTS_DIR, 'energy_load_time_series.png')
plt.savefig(energy_ts_path, dpi=300)
plt.close()

# Plot a combined view with news indicators
if feature_categories['news']:
    plt.figure(figsize=(16, 10))
    
    # Plot target
    ax1 = plt.subplot(211)
    ax1.plot(sample_df['datetime'], sample_df[target], 'b-', label=target)
    ax1.set_ylabel('Power Demand', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Plot top news indicators on second y-axis
    ax2 = ax1.twinx()
    for i, feature in enumerate(feature_categories['news'][:3]):  # Top 3 news features
        color = f'C{i+1}'
        ax2.plot(sample_df['datetime'], sample_df[feature], color=color, alpha=0.7, label=feature)
    
    ax2.set_ylabel('News Indicators', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Energy Load vs. Top News Indicators')
    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, 'energy_load_vs_news_indicators.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()

# Feature selection with optimized approach
logger.info("Performing feature selection...")
def select_features(correlation_with_target, threshold=0.15, max_features=30):
    # Get features with correlation above threshold
    features_above_threshold = correlation_with_target[
        (correlation_with_target.abs() > threshold) & 
        (correlation_with_target.index != target)
    ].index.tolist()
    
    # Cap at max_features (take highest correlation ones)
    if len(features_above_threshold) > max_features:
        features_above_threshold = correlation_with_target.abs().nlargest(max_features+1).index.tolist()
        if target in features_above_threshold:
            features_above_threshold.remove(target)
    
    return features_above_threshold

# Select features with moderate to strong correlation
selected_features = select_features(correlation_with_target)
logger.info(f"Selected {len(selected_features)} features for prediction model")

# Categorize selected features
selected_weather = [f for f in selected_features if f in feature_categories.get('weather', [])]
selected_news = [f for f in selected_features if f in feature_categories.get('news', [])]
selected_temporal = [f for f in selected_features if f in feature_categories.get('temporal', [])]
selected_other = [f for f in selected_features if f not in selected_weather + selected_news + selected_temporal]

logger.info(f"Selected weather features: {len(selected_weather)}")
logger.info(f"Selected news features: {len(selected_news)}")
logger.info(f"Selected temporal features: {len(selected_temporal)}")
logger.info(f"Selected other features: {len(selected_other)}")

# Save selected features to file with categories
selected_df = pd.DataFrame({
    'feature_name': selected_features,
    'correlation': [correlation_with_target[f] for f in selected_features],
    'category': ['weather' if f in selected_weather else
                'news' if f in selected_news else
                'temporal' if f in selected_temporal else 'other' 
                for f in selected_features]
})

selected_path = os.path.join(RESULTS_DIR, 'selected_features.csv')
selected_df.to_csv(selected_path, index=False)
logger.info(f"Selected features saved to {selected_path}")

logger.info(f"Analysis complete! Results saved to {RESULTS_DIR}")


