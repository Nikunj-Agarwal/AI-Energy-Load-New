import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import configuration - ADDED IMPORT FROM CONFIG FILE
from config import (
    PATHS, setup_and_verify, test_directory_writing as config_test_directory_writing
)

# Use paths from config instead of hardcoded paths - IMPROVED PATH USAGE
INPUT_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")
OUTPUT_PATH = os.path.join(PATHS["AGGREGATED_DIR"], "aggregated_gkg_15min.csv")
FIGURES_DIR = PATHS["FIGURES_DIR"]

# Define key theme categories most likely to affect energy load
# Updated to match categories used in sparsing.py
ENERGY_THEMES = ['Energy', 'Environment', 'Infrastructure', 'Social', 'Health', 
                'Political', 'Economic']  # Added more relevant themes

def validate_input_data(df, required_columns=None):
    """Validate that input data has required columns for aggregation"""
    if required_columns is None:
        required_columns = ['datetime', 'GKGRECORDID'] + [f'theme_{theme}' for theme in ENERGY_THEMES]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # Try case-insensitive matching for theme columns
        theme_cols = [col for col in missing_columns if col.startswith('theme_')]
        for theme_col in theme_cols[:]:  # Use copy to avoid modification during iteration
            # Try different capitalizations
            theme_name = theme_col.replace('theme_', '')
            alternatives = [
                f'theme_{theme_name.lower()}',
                f'theme_{theme_name.upper()}',
                f'theme_{theme_name.capitalize()}'
            ]
            for alt in alternatives:
                if alt in df.columns:
                    print(f"Found alternative column {alt} for {theme_col}")
                    # Create the expected column
                    df[theme_col] = df[alt]
                    missing_columns.remove(theme_col)
                    break
        
        if missing_columns:
            print(f"WARNING: Input data missing required columns: {missing_columns}")
            return False
    
    return True

# Update the aggregate_chunk function to handle new features

def aggregate_chunk(chunk):
    """Aggregate a single chunk of data into 15-minute intervals"""
    # Ensure datetime is in the correct format
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    
    # Create 15-minute time buckets
    chunk['time_bucket'] = chunk['datetime'].dt.floor('15min')
    
    # Define aggregation dictionary with enhanced features
    agg_dict = {
        'GKGRECORDID': 'count',
        **{f'theme_{cat}': ['sum', 'max', 'mean'] for cat in ENERGY_THEMES},
        'tone_tone': ['mean', 'min', 'max', 'std'],
        'tone_negative': ['max', 'mean'],
        'tone_positive': ['max', 'mean'],
        'tone_polarity': ['mean', 'max'],
        'tone_activity': ['mean', 'max'],
        
        # Use entity data if available but don't rely on it (high missing rate)
        'entity_count': ['sum', 'mean'] if 'entity_count' in chunk.columns else [],
        'entity_variety': ['sum', 'max'] if 'entity_variety' in chunk.columns else [],
        
        # Amount fields have better coverage than counts (only ~20% missing)
        'avg_amount': ['mean', 'max', 'sum'] if 'avg_amount' in chunk.columns else [],
        'max_amount': ['max', 'mean'] if 'max_amount' in chunk.columns else [],
        'amount_count': ['sum'] if 'amount_count' in chunk.columns else [],
        
        # Add the new impact score
        'energy_impact_score': ['mean', 'max', 'sum'] if 'energy_impact_score' in chunk.columns else [],
        
        # Add energy context features
        **{context: ['sum', 'max'] for context in [
            'energy_supply', 'energy_demand', 'energy_price', 
            'energy_policy', 'energy_infrastructure', 
            'renewable_energy', 'fossil_fuel', 'weather_event'
        ] if context in chunk.columns},
        
        # Include amount-theme interaction features
        'energy_amount': ['sum', 'max'] if 'energy_amount' in chunk.columns else [],
        'infrastructure_amount': ['sum', 'max'] if 'infrastructure_amount' in chunk.columns else [],
        'environment_amount': ['sum', 'max'] if 'environment_amount' in chunk.columns else []
    }
    
    # Filter to only include columns that exist in the dataframe
    agg_dict = {k: v for k, v in agg_dict.items() if isinstance(v, list) and k in chunk.columns}
    
    # Group by time bucket and aggregate
    try:
        agg_chunk = chunk.groupby('time_bucket').agg(agg_dict)
        
        # Flatten column multi-index
        agg_chunk.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_chunk.columns]
        
        # Ensure article_count exists, creating it if needed
        if 'GKGRECORDID_count' in agg_chunk.columns:
            agg_chunk.rename(columns={'GKGRECORDID_count': 'article_count'}, inplace=True)
        elif 'article_count' not in agg_chunk.columns:
            # Create article_count by counting rows directly
            article_counts = chunk.groupby('time_bucket').size()
            agg_chunk['article_count'] = article_counts
        
        return agg_chunk
    except Exception as e:
        print(f"ERROR during chunk aggregation: {e}")
        # Try with minimal set of columns
        try:
            minimal_agg = {
                'GKGRECORDID': 'count',
                'tone_tone': ['mean'] if 'tone_tone' in chunk.columns else []
            }
            minimal_agg = {k: v for k, v in minimal_agg.items() if k in chunk.columns}
            print("Retrying with minimal aggregation...")
            agg_chunk = chunk.groupby('time_bucket').agg(minimal_agg)
            agg_chunk.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_chunk.columns]
            
            # Ensure article_count always exists
            if 'GKGRECORDID_count' in agg_chunk.columns:
                agg_chunk.rename(columns={'GKGRECORDID_count': 'article_count'}, inplace=True)
            else:
                # Create article_count by counting rows directly
                article_counts = chunk.groupby('time_bucket').size()
                agg_chunk['article_count'] = article_counts
                
            return agg_chunk
        except Exception as e2:
            print(f"ERROR: Minimal aggregation also failed: {e2}")
            return None

# Update the engineer_features function to improve feature creation
def engineer_features(agg_df):
    """Add engineered features to the aggregated dataframe"""
    if agg_df is None or len(agg_df) == 0:
        return agg_df
        
    # Reset index to make time_bucket a column
    agg_df = agg_df.reset_index()
    
    # Ensure article_count exists
    if 'article_count' not in agg_df.columns:
        print("WARNING: article_count column not found, creating placeholder")
        agg_df['article_count'] = 1
    
    # Calculate tone volatility
    agg_df['tone_volatility'] = agg_df['tone_tone_max'] - agg_df['tone_tone_min'] \
                            if all(col in agg_df.columns for col in ['tone_tone_max', 'tone_tone_min']) \
                            else 0
    
    # Enhanced energy-specific features using the new columns
    
    # 1. Create better energy crisis indicators using amount data
    if all(col in agg_df.columns for col in ['theme_Energy_sum', 'avg_amount_sum']):
        # Energy event magnitude (using more reliable amount data)
        agg_df['energy_event_magnitude'] = (
            agg_df['theme_Energy_sum'] * 
            np.log1p(agg_df['avg_amount_sum'])
        )
    else:
        # Fallback if amount data isn't available
        agg_df['energy_event_magnitude'] = (
            agg_df['theme_Energy_sum'] * 
            agg_df['article_count']
        )
    
    # 2. Use energy impact score if available
    if 'energy_impact_score_sum' in agg_df.columns:
        agg_df['energy_impact_index'] = agg_df['energy_impact_score_sum']
    
    # 3. Refined crisis indicators using energy contexts
    energy_contexts = [
        'energy_supply', 'energy_demand', 'energy_price', 
        'energy_infrastructure', 'weather_event'
    ]
    
    for context in energy_contexts:
        sum_col = f'{context}_sum'
        if sum_col in agg_df.columns:
            # Create context-specific indicators
            if context == 'energy_supply':
                agg_df['supply_disruption_indicator'] = (
                    agg_df[sum_col] * agg_df['tone_negative_max']
                )
            elif context == 'energy_demand':
                agg_df['demand_spike_indicator'] = (
                    agg_df[sum_col] * agg_df['article_count']
                )
            elif context == 'energy_price':
                agg_df['price_volatility_indicator'] = (
                    agg_df[sum_col] * agg_df['tone_volatility']
                )
            elif context == 'energy_infrastructure':
                agg_df['infrastructure_failure_indicator'] = (
                    agg_df[sum_col] * agg_df['tone_negative_max']
                )
            elif context == 'weather_event':
                agg_df['weather_impact_indicator'] = (
                    agg_df[sum_col] * agg_df['article_volume_spike']
                    if 'article_volume_spike' in agg_df.columns else agg_df[sum_col]
                )
    
    # Existing composite features with updates to handle missing columns better
    feature_definitions = [
        ('energy_crisis_indicator', 
         lambda df: df['theme_Energy_sum'] * df['tone_negative_max'] 
         if all(col in df.columns for col in ['theme_Energy_sum', 'tone_negative_max']) else 
         df['theme_Energy_sum'] * 0.5  # Fallback if tone data missing
         if 'theme_Energy_sum' in df.columns else 0),
         
        ('weather_alert_indicator', 
         lambda df: df['theme_Environment_sum'] * np.abs(df['tone_tone_min'])
         if all(col in df.columns for col in ['theme_Environment_sum', 'tone_tone_min']) else 
         df['theme_Environment_sum'] * 0.5  # Fallback
         if 'theme_Environment_sum' in df.columns else 0),
         
        ('social_event_indicator', 
         lambda df: df['theme_Social_sum'] * df['article_count'] / 100
         if all(col in df.columns for col in ['theme_Social_sum', 'article_count']) else 0),
         
        ('infrastructure_stress', 
         lambda df: df['theme_Infrastructure_sum'] * df['tone_negative_max']
         if all(col in df.columns for col in ['theme_Infrastructure_sum', 'tone_negative_max']) else 0),
         
        ('political_crisis_indicator', 
         lambda df: df['theme_Political_sum'] * df['tone_negative_max']
         if all(col in df.columns for col in ['theme_Political_sum', 'tone_negative_max']) else 0),
         
        ('economic_impact_indicator', 
         lambda df: df['theme_Economic_sum'] * df['tone_volatility']
         if all(col in df.columns for col in ['theme_Economic_sum', 'tone_volatility']) else 0)
    ]
    
    # Apply each feature definition with try/except
    for feature_name, feature_func in feature_definitions:
        try:
            agg_df[feature_name] = feature_func(agg_df)
        except Exception as e:
            print(f"WARNING: Could not create feature {feature_name}: {e}")
            agg_df[feature_name] = 0
    
    # Article volume changes
    try:
        agg_df['prev_article_count'] = agg_df['article_count'].shift(1).fillna(0)
        agg_df['article_count_change'] = agg_df['article_count'] - agg_df['prev_article_count']
        
        # Use try/except for rolling window operations
        try:
            agg_df['article_volume_spike'] = (agg_df['article_count'] > 
                                          agg_df['article_count'].rolling(window=12).mean() * 1.5).astype(int)
        except Exception as e:
            print(f"WARNING: Could not calculate article_volume_spike: {e}")
            agg_df['article_volume_spike'] = 0
            
    except Exception as e:
        print(f"WARNING: Could not calculate article volume changes: {e}")
        agg_df['article_count_change'] = 0
        agg_df['article_volume_spike'] = 0
    
    # Add time features
    agg_df['hour'] = agg_df['time_bucket'].dt.hour
    agg_df['day_of_week'] = agg_df['time_bucket'].dt.dayofweek
    agg_df['is_weekend'] = agg_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    agg_df['is_business_hours'] = ((agg_df['hour'] >= 9) & (agg_df['hour'] <= 17) & 
                                 (agg_df['is_weekend'] == 0)).astype(int)
    agg_df['month'] = agg_df['time_bucket'].dt.month
    agg_df['day'] = agg_df['time_bucket'].dt.day
    
    # Seasonality features
    agg_df['hour_sin'] = np.sin(2 * np.pi * agg_df['hour'] / 24)
    agg_df['hour_cos'] = np.cos(2 * np.pi * agg_df['hour'] / 24)
    agg_df['day_of_week_sin'] = np.sin(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['day_of_week_cos'] = np.cos(2 * np.pi * agg_df['day_of_week'] / 7)
    
    return agg_df

def select_features(df):
    """Select and filter the most relevant features"""
    # List of important features to keep
    keep_features = [
        'time_bucket', 'article_count', 'article_count_change', 'article_volume_spike',
        
        # Core tone metrics
        'tone_tone_mean', 'tone_negative_max', 'tone_positive_max', 'tone_volatility',
        'tone_polarity_mean', 'tone_activity_mean',
        
        # Theme sums
        'theme_Energy_sum', 'theme_Environment_sum', 'theme_Infrastructure_sum', 
        'theme_Social_sum', 'theme_Health_sum', 'theme_Political_sum', 'theme_Economic_sum',
        
        # Theme intensity (max values)
        'theme_Energy_max', 'theme_Environment_max', 'theme_Infrastructure_max',
        
        # Entity and amount metrics
        'entity_count_sum', 'entity_variety_max', 'max_amount_max',
        
        # Composite indicators
        'energy_crisis_indicator', 'weather_alert_indicator', 
        'social_event_indicator', 'infrastructure_stress',
        'political_crisis_indicator', 'economic_impact_indicator',
        
        # Time features
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'month', 'day', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
    ]
    
    # Ensure all requested columns exist
    available_columns = df.columns.tolist()
    final_columns = [col for col in keep_features if col in available_columns]
    
    if len(final_columns) < len(keep_features):
        missing = set(keep_features) - set(final_columns)
        print(f"Warning: Some columns were not found in the data: {missing}")
    
    # Keep only the most relevant features
    return df[final_columns]

def handle_missing_intervals(df, start_date=None, end_date=None, freq='15min'):
    """
    Ensure all intervals exist in the time range, filling gaps with appropriate values.
    This is important for time series analysis.
    """
    print("Checking for missing time intervals...")
    
    # If no dates provided, use min/max from the data
    if start_date is None:
        start_date = df['time_bucket'].min()
    if end_date is None:
        end_date = df['time_bucket'].max()
    
    print(f"Time range: {start_date} to {end_date}")
    
    # Create complete interval range
    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    if len(complete_range) == len(df):
        print("No missing intervals found.")
    else:
        print(f"Found {len(complete_range) - len(df)} missing intervals out of {len(complete_range)} total.")
    
    # Create a reference dataframe with all intervals
    ref_df = pd.DataFrame({'time_bucket': complete_range})
    
    # Merge with actual data
    merged_df = pd.merge(ref_df, df, on='time_bucket', how='left')
    
    # Fill missing values with appropriate values
    # For article count and most sum metrics, use 0
    sum_cols = [col for col in merged_df.columns if col.endswith('_sum') or col == 'article_count']
    merged_df[sum_cols] = merged_df[sum_cols].fillna(0)
    
    # For means and other metrics, use forward fill then backward fill
    other_numeric = merged_df.select_dtypes(include=[np.number]).columns.difference(sum_cols)
    merged_df[other_numeric] = merged_df[other_numeric].fillna(method='ffill')
    merged_df[other_numeric] = merged_df[other_numeric].fillna(method='bfill')
    
    # Any remaining missing values fill with 0
    merged_df = merged_df.fillna(0)
    
    return merged_df

def save_feature_documentation(df, output_dir):
    """Save feature statistics and documentation"""
    stats_file = os.path.join(output_dir, "feature_statistics.csv")
    
    # Calculate basic statistics
    feature_stats = pd.DataFrame({
        'mean': df.mean(),
        'min': df.min(),
        'max': df.max(),
        'std': df.std(),
        'missing': df.isnull().sum()
    })
    
    # Save to CSV
    feature_stats.to_csv(stats_file, encoding='utf-8')
    print(f"Feature statistics saved to {stats_file}")

def process_file_in_chunks(input_path, output_path, chunk_size=50000, start_date=None, end_date=None):
    """Process a large file in chunks with minimal memory usage"""
    print(f"Processing file {input_path} in chunks of {chunk_size} rows...")
    
    # Dictionary to store aggregated results by time bucket
    aggregated_data = {}
    
    # Read and process the file in chunks
    chunks_processed = 0
    total_rows = 0
    
    try:
        # Initialize the reader - this doesn't load the whole file at once
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, encoding='utf-8', encoding_errors='replace'):
            chunks_processed += 1
            total_rows += len(chunk)
            print(f"Processing chunk {chunks_processed} ({len(chunk)} rows)...")
            
            # Validate data in chunk
            if not validate_input_data(chunk):
                print(f"WARNING: Chunk {chunks_processed} has invalid data structure. Skipping.")
                continue
            
            # Aggregate this chunk
            chunk_agg = aggregate_chunk(chunk)
            
            if chunk_agg is None or len(chunk_agg) == 0:
                print(f"WARNING: Chunk {chunks_processed} produced no aggregated data. Skipping.")
                continue
            
            # Add to our aggregated data dictionary
            for idx, row in chunk_agg.iterrows():
                time_bucket = idx  # The index is the time_bucket
                
                if time_bucket in aggregated_data:
                    # Merge with existing data for this time bucket
                    for col in chunk_agg.columns:
                        # For count/sum metrics, add values
                        if col.endswith('_sum') or col == 'article_count':
                            aggregated_data[time_bucket][col] = aggregated_data[time_bucket].get(col, 0) + row[col]
                        # For max metrics, take the maximum
                        elif col.endswith('_max'):
                            aggregated_data[time_bucket][col] = max(aggregated_data[time_bucket].get(col, -np.inf), row[col])
                        # For mean metrics, we'll need to recalculate later
                        elif col.endswith('_mean'):
                            # Store sum and count for later mean calculation
                            sum_col = f"{col}_sum"
                            count_col = f"{col}_count"
                            aggregated_data[time_bucket][sum_col] = aggregated_data[time_bucket].get(sum_col, 0) + row[col]
                            aggregated_data[time_bucket][count_col] = aggregated_data[time_bucket].get(count_col, 0) + 1
                        # For min metrics, take the minimum
                        elif col.endswith('_min'):
                            aggregated_data[time_bucket][col] = min(aggregated_data[time_bucket].get(col, np.inf), row[col])
                        # For std metrics, we will need raw data which we don't have, so skip
                else:
                    # New time bucket
                    aggregated_data[time_bucket] = row.to_dict()
                    # Initialize count for mean metrics
                    for col in row.index:
                        if col.endswith('_mean'):
                            aggregated_data[time_bucket][f"{col}_count"] = 1
                            aggregated_data[time_bucket][f"{col}_sum"] = row[col]
            
            # Clear memory
            del chunk
            del chunk_agg
            
    except Exception as e:
        print(f"Error processing chunks: {e}")
        # Try with latin-1 encoding if UTF-8 failed
        if 'UTF-8' in str(e) or 'encoding' in str(e).lower():
            print("Retrying with latin-1 encoding...")
            return process_file_in_chunks_latin1(input_path, output_path, chunk_size, start_date, end_date)
    
    print(f"Processed {chunks_processed} chunks with {total_rows} total rows")
    print(f"Found {len(aggregated_data)} unique time buckets")
    
    # Convert aggregated data to DataFrame
    agg_df = pd.DataFrame.from_dict(aggregated_data, orient='index')
    agg_df.index.name = 'time_bucket'
    
    # Recalculate mean values
    for col in list(agg_df.columns):
        if col.endswith('_mean'):
            sum_col = f"{col}_sum"
            count_col = f"{col}_count"
            if sum_col in agg_df.columns and count_col in agg_df.columns:
                agg_df[col] = agg_df[sum_col] / agg_df[count_col]
                # Drop temporary columns
                agg_df.drop([sum_col, count_col], axis=1, inplace=True)
    
    # Engineer features
    print("Engineering features...")
    agg_df = engineer_features(agg_df)
    
    # Select only needed features
    print("Selecting relevant features...")
    final_df = select_features(agg_df)
    
    # Handle missing intervals
    print("Handling missing intervals...")
    complete_df = handle_missing_intervals(final_df, start_date, end_date)
    
    # Save to file
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    complete_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Saved {len(complete_df)} rows with {complete_df.shape[1]} columns")
    
    return complete_df

def process_file_in_chunks_latin1(input_path, output_path, chunk_size=50000, start_date=None, end_date=None):
    """Process a large file in chunks with latin-1 encoding (fallback)"""
    # Same as process_file_in_chunks but with latin-1 encoding
    # I'll provide the implementation with latin-1 encoding
    try:
        chunks_processed = 0
        total_rows = 0
        aggregated_data = {}
        
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, encoding='latin-1'):
            # Same processing as in the main function
            chunks_processed += 1
            total_rows += len(chunk)
            print(f"Processing chunk {chunks_processed} ({len(chunk)} rows) with latin-1 encoding...")
            
            # Validate data in chunk
            if not validate_input_data(chunk):
                print(f"WARNING: Chunk {chunks_processed} has invalid data structure. Skipping.")
                continue
            
            # Aggregate this chunk
            chunk_agg = aggregate_chunk(chunk)
            
            if chunk_agg is None or len(chunk_agg) == 0:
                print(f"WARNING: Chunk {chunks_processed} produced no aggregated data. Skipping.")
                continue
            
            # Add to our aggregated data dictionary
            for idx, row in chunk_agg.iterrows():
                time_bucket = idx  # The index is the time_bucket
                
                if time_bucket in aggregated_data:
                    # Merge with existing data for this time bucket
                    for col in chunk_agg.columns:
                        # For count/sum metrics, add values
                        if col.endswith('_sum') or col == 'article_count':
                            aggregated_data[time_bucket][col] = aggregated_data[time_bucket].get(col, 0) + row[col]
                        # For max metrics, take the maximum
                        elif col.endswith('_max'):
                            aggregated_data[time_bucket][col] = max(aggregated_data[time_bucket].get(col, -np.inf), row[col])
                        # For mean metrics, we'll need to recalculate later
                        elif col.endswith('_mean'):
                            # Store sum and count for later mean calculation
                            sum_col = f"{col}_sum"
                            count_col = f"{col}_count"
                            aggregated_data[time_bucket][sum_col] = aggregated_data[time_bucket].get(sum_col, 0) + row[col]
                            aggregated_data[time_bucket][count_col] = aggregated_data[time_bucket].get(count_col, 0) + 1
                        # For min metrics, take the minimum
                        elif col.endswith('_min'):
                            aggregated_data[time_bucket][col] = min(aggregated_data[time_bucket].get(col, np.inf), row[col])
                        # For std metrics, we will need raw data which we don't have, so skip
                else:
                    # New time bucket
                    aggregated_data[time_bucket] = row.to_dict()
                    # Initialize count for mean metrics
                    for col in row.index:
                        if col.endswith('_mean'):
                            aggregated_data[time_bucket][f"{col}_count"] = 1
                            aggregated_data[time_bucket][f"{col}_sum"] = row[col]
            
            # Clear memory
            del chunk
            del chunk_agg
            
        # Continue with the same processing as in the main function
        print(f"Processed {chunks_processed} chunks with {total_rows} total rows")
        print(f"Found {len(aggregated_data)} unique time buckets")
        
        # Convert aggregated data to DataFrame
        agg_df = pd.DataFrame.from_dict(aggregated_data, orient='index')
        agg_df.index.name = 'time_bucket'
        
        # Recalculate mean values
        for col in list(agg_df.columns):
            if col.endswith('_mean'):
                sum_col = f"{col}_sum"
                count_col = f"{col}_count"
                if sum_col in agg_df.columns and count_col in agg_df.columns:
                    agg_df[col] = agg_df[sum_col] / agg_df[count_col]
                    # Drop temporary columns
                    agg_df.drop([sum_col, count_col], axis=1, inplace=True)
        
        # Engineer features
        print("Engineering features...")
        agg_df = engineer_features(agg_df)
        
        # Select only needed features
        print("Selecting relevant features...")
        final_df = select_features(agg_df)
        
        # Handle missing intervals
        print("Handling missing intervals...")
        complete_df = handle_missing_intervals(final_df, start_date, end_date)
        
        # Save to file
        print(f"Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        complete_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Saved {len(complete_df)} rows with {complete_df.shape[1]} columns")
        
        return complete_df
        
    except Exception as e:
        print(f"Fatal error during latin-1 processing: {e}")
        return None

def analyze_features(df):
    """
    Analyze feature importance and correlations to help with feature selection.
    """
    print("Analyzing feature relationships...")
    
    # Skip non-numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_correlations.png'), dpi=300)
    plt.close()
    
    # Print highly correlated features
    print("\nHighly correlated features (r > 0.8):")
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if high_corr.iloc[i, j]:
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    for col1, col2, corr in sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"{col1} & {col2}: {corr:.3f}")
    
    # Feature distribution analysis
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(ENERGY_THEMES[:5]):  # First 5 themes
        plt.subplot(2, 3, i+1)
        sns.histplot(df[f'theme_{col}_sum'], kde=True)
        plt.title(f'{col} Theme Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'theme_distributions.png'), dpi=300)
    plt.close()
    
    return corr_matrix

def generate_time_series_plots(df):
    """Generate exploratory time series plots of key features"""
    print("Generating time series visualizations...")
    
    # Ensure required columns exist
    required_cols = ['time_bucket', 'article_count', 'article_volume_spike']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Cannot generate all plots. Missing columns: {missing_cols}")
        # Create dummy columns for missing data
        for col in missing_cols:
            if col == 'article_count':
                df[col] = 1  # Default value
            elif col == 'article_volume_spike':
                df[col] = 0  # Default value
    
    # Select a subset of data points for clarity if dataset is large
    plot_df = df
    if len(df) > 1000:
        # Sample at 1-hour intervals for cleaner plots
        plot_df = df.iloc[::4]  
    
    # 1. Article volume and spikes - with column existence checks
    plt.figure(figsize=(15, 8))
    plt.plot(plot_df['time_bucket'], plot_df['article_count'], label='Article Count')
    
    if 'article_volume_spike' in plot_df.columns:
        plt.scatter(plot_df['time_bucket'][plot_df['article_volume_spike'] == 1], 
                    plot_df['article_count'][plot_df['article_volume_spike'] == 1],
                    color='red', alpha=0.6, label='Volume Spikes')
    
    plt.title('Article Volume Over Time with Detected Spikes', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Article Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'article_volume_spikes.png'), dpi=300)
    plt.close()
    
    # Continue with other visualization code, adding similar checks
    
    # 2. Theme presence over time
    plt.figure(figsize=(15, 10))
    for i, theme in enumerate(ENERGY_THEMES[:5]):  # First 5 themes
        plt.plot(plot_df['time_bucket'], plot_df[f'theme_{theme}_sum'], 
                label=f'{theme} Theme', alpha=0.7)
    plt.title('Theme Presence Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Theme Mentions', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'theme_presence.png'), dpi=300)
    plt.close()
    
    # 3. Crisis indicators
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(plot_df['time_bucket'], plot_df['energy_crisis_indicator'], 
             label='Energy Crisis', color='red', alpha=0.7)
    plt.plot(plot_df['time_bucket'], plot_df['infrastructure_stress'], 
             label='Infrastructure Stress', color='orange', alpha=0.7)
    plt.title('Energy & Infrastructure Crisis Indicators', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(plot_df['time_bucket'], plot_df['weather_alert_indicator'], 
             label='Weather Alert', color='blue', alpha=0.7)
    plt.plot(plot_df['time_bucket'], plot_df['social_event_indicator'], 
             label='Social Event', color='green', alpha=0.7)
    plt.title('Weather & Social Event Indicators', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'crisis_indicators.png'), dpi=300)
    plt.close()
    
    # 4. Tone metrics
    plt.figure(figsize=(15, 8))
    plt.plot(plot_df['time_bucket'], plot_df['tone_tone_mean'], label='Average Tone', color='purple')
    plt.fill_between(plot_df['time_bucket'], 
                     plot_df['tone_tone_mean'] - plot_df['tone_volatility']/2,
                     plot_df['tone_tone_mean'] + plot_df['tone_volatility']/2, 
                     alpha=0.3, color='purple', label='Tone Volatility')
    plt.title('Sentiment Tone and Volatility Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tone_metrics.png'), dpi=300)
    plt.close()
    
    # 5. Weekly patterns
    daily_pattern = df.groupby('hour')['article_count'].mean().reset_index()
    weekly_pattern = df.groupby('day_of_week')['article_count'].mean().reset_index()
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='hour', y='article_count', data=daily_pattern)
    plt.title('Average Article Volume by Hour of Day', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Article Count', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='day_of_week', y='article_count', data=weekly_pattern)
    plt.title('Average Article Volume by Day of Week', fontsize=14)
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    plt.ylabel('Average Article Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_patterns.png'), dpi=300)
    plt.close()

def test_directory_writing():
    """Test writing to all required directories before processing"""
    print("Testing directory writing permissions...")
    
    # IMPROVED DIRECTORY TESTING - More comprehensive tests
    directories = [
        os.path.dirname(INPUT_PATH),  # Added input directory
        os.path.dirname(OUTPUT_PATH),
        FIGURES_DIR
    ]
    
    all_passed = True
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"ERROR: Could not create directory {directory}: {e}")
                all_passed = False
                continue
        
        # Test writing a small file
        test_file = os.path.join(directory, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test write at {datetime.now()}")
            os.remove(test_file)
            print(f"✓ Successfully wrote to {directory}")
        except Exception as e:
            print(f"✗ ERROR: Could not write to {directory}: {e}")
            all_passed = False
    
    # Test writing a DataFrame to output directory
    try:
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        test_csv_path = os.path.join(os.path.dirname(OUTPUT_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv_path, index=False)
        os.remove(test_csv_path)
        print(f"✓ Successfully wrote test DataFrame to {os.path.dirname(OUTPUT_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write test DataFrame: {e}")
        all_passed = False
            
    # Also test writing a small test plot to ensure matplotlib can save figures
    if os.path.exists(FIGURES_DIR):
        try:
            test_fig_path = os.path.join(FIGURES_DIR, "test_plot.png")
            plt.figure(figsize=(2, 2))
            plt.plot([1, 2, 3], [1, 4, 9])
            plt.savefig(test_fig_path)
            plt.close()
            os.remove(test_fig_path)
            print(f"✓ Successfully created test plot in {FIGURES_DIR}")
        except Exception as e:
            print(f"✗ ERROR: Could not create test plot: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main function with memory-efficient processing"""
    print(f"=== GDELT GKG Data Aggregation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="GDELT GKG Data Aggregation")
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for processing")
    args = parser.parse_args()
    
    # Override paths if specified
    input_path = args.input or INPUT_PATH
    output_path = args.output or OUTPUT_PATH
    chunk_size = args.chunk_size
    
    # Directory checks
    if not setup_and_verify() or not test_directory_writing():
        print("ERROR: Directory checks failed. Aborting processing.")
        return
    
    # Parse date parameters if provided
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    
    # Process the file in chunks
    print(f"Starting memory-efficient processing with chunk size: {chunk_size}")
    complete_df = process_file_in_chunks(input_path, output_path, chunk_size, start_date, end_date)
    
    if complete_df is None:
        print("Processing failed. See errors above.")
        return
    
    # Only generate visualizations if not disabled
    if not args.no_plots:
        # Analyze features
        analyze_features(complete_df)
        
        # Generate exploratory visualizations
        generate_time_series_plots(complete_df)
    
    # Save feature documentation
    save_feature_documentation(complete_df, os.path.dirname(output_path))
    
    # Print success message
    print(f"Data aggregation complete. Created {len(complete_df)} 15-minute intervals with {complete_df.shape[1]} features.")
    
    return complete_df

if __name__ == "__main__":
    main()