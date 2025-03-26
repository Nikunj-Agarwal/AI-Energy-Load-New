import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to ensure we can import config properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import (
    PATHS, THEME_CATEGORIES, ENERGY_THEMES, 
    OUTPUT_DIR, FEATURE_ENGINEERING
)

def add_exponential_decay_features(df, theme_cols=None, decay_rates=None):
    """
    Add exponential decay features to model how theme impact diminishes over time
    
    Args:
        df: DataFrame with time-ordered theme data
        theme_cols: List of theme columns to process, or None for all theme columns
        decay_rates: Dictionary of decay rates or None to use defaults
        
    Returns:
        DataFrame with added decay features
    """
    print("Adding exponential decay features...")
    
    # Ensure we have time-sorted data
    if 'time_bucket' not in df.columns:
        print("No time_bucket column found. Cannot calculate decay effects.")
        return df
    
    # Ensure time_bucket is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_bucket']):
        df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    
    # Sort by time
    df = df.sort_values('time_bucket')
    
    # Set default decay rates if not provided
    if decay_rates is None:
        decay_rates = {
            'fast': 0.5,    # Fast decay (impact halves each period)
            'medium': 0.2,  # Medium decay
            'slow': 0.05    # Slow decay (long-lasting effects)
        }
    
    # Get theme columns if not specified
    if theme_cols is None:
        theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Generate lag periods with different decay rates
    lag_periods = [1, 2, 4, 8, 12, 24]  # 15min intervals: 15m, 30m, 1h, 2h, 3h, 6h
    
    # Dictionary to store all new features
    new_features = {}
    
    # Calculate lags with decay for each theme and decay rate
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        for lag in lag_periods:
            # Create basic lag feature
            lag_col_name = f'{base_name}_lag{lag}'
            new_features[lag_col_name] = df[col].shift(lag).fillna(0)
            
            # Apply different decay rates
            for decay_name, decay_rate in decay_rates.items():
                decay_factor = np.exp(-decay_rate * lag)
                decay_col_name = f'{base_name}_lag{lag}_{decay_name}_decay'
                new_features[decay_col_name] = new_features[lag_col_name] * decay_factor
    
    # Special handling for energy-related themes
    energy_theme_cols = [col for col in theme_cols if 'Energy' in col or 'energy' in col]
    
    if energy_theme_cols:
        # Create cumulative decay features for energy themes
        for col in energy_theme_cols:
            base_name = col.replace('_sum', '')
            
            # Calculate exponentially weighted moving averages with different spans
            # These represent accumulated impact with natural decay
            new_features[f'{base_name}_ewma_fast'] = df[col].ewm(span=4).mean()    # ~1 hour memory
            new_features[f'{base_name}_ewma_medium'] = df[col].ewm(span=24).mean() # ~6 hour memory
            new_features[f'{base_name}_ewma_slow'] = df[col].ewm(span=96).mean()   # ~24 hour memory
    
    # Add all new features at once to avoid fragmentation
    result_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    
    return result_df

def create_temporal_theme_windows(df, window_sizes=[4, 8, 16, 24]):
    """
    Create sliding window features that capture theme presence over time windows
    
    Args:
        df: DataFrame with time-ordered theme data
        window_sizes: List of window sizes to use
        
    Returns:
        DataFrame with added window features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Create window features
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        for window in window_sizes:
            # Sum over window (total mentions in window)
            df[f'{base_name}_window{window}_sum'] = df[col].rolling(window=window, min_periods=1).sum()
            
            # Mean over window (average mentions per period)
            df[f'{base_name}_window{window}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Maximum over window (peak intensity)
            df[f'{base_name}_window{window}_max'] = df[col].rolling(window=window, min_periods=1).max()
            
            # Minimum over window (baseline level)
            df[f'{base_name}_window{window}_min'] = df[col].rolling(window=window, min_periods=1).min()
            
            # Standard deviation (volatility in window)
            df[f'{base_name}_window{window}_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    
    return df

def add_time_weighted_features(df, theme_cols=None):
    """Add weighted features based on time of day and week"""
    if 'hour' not in df.columns or 'is_weekend' not in df.columns:
        print("Required time columns missing, adding them first...")
        df['hour'] = df['time_bucket'].dt.hour
        df['day_of_week'] = df['time_bucket'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                   (df['is_weekend'] == 0)).astype(int)
    
    # Determine business hours weight (higher during business hours)
    business_hours_weight = df['is_business_hours'] * 0.5 + 0.5
    
    # Weekend weight (higher on weekends for residential impact)
    weekend_weight = df['is_weekend'] * 0.3 + 0.7
    
    # Evening peak hours (5pm-9pm)
    evening_peak = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    evening_peak_weight = evening_peak * 0.4 + 0.6
    
    # Combined weight
    combined_weight = (business_hours_weight + weekend_weight + evening_peak_weight) / 3
    
    # Process theme columns
    if theme_cols is None:
        theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Create all features at once
    new_features = {}
    
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        # Create all the weighted features
        new_features[f'{base_name}_business_impact'] = df[col] * business_hours_weight
        new_features[f'{base_name}_weekend_adjusted'] = df[col] * weekend_weight
        new_features[f'{base_name}_evening_impact'] = df[col] * evening_peak_weight
        new_features[f'{base_name}_temporal_weighted'] = df[col] * combined_weight
    
    # Add all new features at once
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    
    return df

def add_temporal_interaction_features(df):
    """
    Add features capturing interaction between themes and temporal factors
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with added interaction features
    """
    # Check if we have proper datetime components
    if 'hour' not in df.columns or 'day_of_week' not in df.columns:
        if 'time_bucket' in df.columns:
            # Extract time components
            df['hour'] = df['time_bucket'].dt.hour
            df['day_of_week'] = df['time_bucket'].dt.dayofweek
        else:
            return df
    
    # Create time period indicators
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_workday'] = (df['day_of_week'] < 5).astype(int)
    
    # Get energy-related theme columns
    energy_cols = [col for col in df.columns if col.startswith('theme_') and 
                  ('Energy' in col or 'energy' in col) and col.endswith('_sum')]
    
    # Create interaction features for energy themes
    for col in energy_cols:
        base_name = col.replace('_sum', '')
        
        # Time of day interactions
        df[f'{base_name}_morning_effect'] = df[col] * df['is_morning']
        df[f'{base_name}_afternoon_effect'] = df[col] * df['is_afternoon']
        df[f'{base_name}_evening_effect'] = df[col] * df['is_evening']
        df[f'{base_name}_night_effect'] = df[col] * df['is_night']
        
        # Day type interactions
        df[f'{base_name}_weekend_effect'] = df[col] * df['is_weekend']
        df[f'{base_name}_workday_effect'] = df[col] * df['is_workday']
        
        # Combined effects
        df[f'{base_name}_evening_workday'] = df[col] * df['is_evening'] * df['is_workday']
        df[f'{base_name}_morning_workday'] = df[col] * df['is_morning'] * df['is_workday']
    
    return df

def create_memory_decay_features(df):
    """
    Create features that model memory effects - how past events continue to influence
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with memory decay features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get tone columns for interaction with themes
    tone_cols = [col for col in df.columns if col.startswith('tone_')]
    
    # Get key theme columns
    energy_cols = [col for col in df.columns if col.startswith('theme_') and 
                  ('Energy' in col or 'energy' in col) and col.endswith('_sum')]
    
    # Dictionary to store all new features
    new_features = {}
    
    # Memory parameters
    memory_alpha = 0.1  # Memory retention factor (smaller = longer memory)
    
    # Create memory features for key themes
    for col in energy_cols:
        base_name = col.replace('_sum', '')
        
        # Memory effect has higher impact if tone was negative
        if 'tone_negative_max' in df.columns:
            # Enhanced memory for negative events
            negative_events = (df[col] > 0) & (df['tone_negative_max'] > 0.5)
            
            # Calculate cumulative memory effect with decay
            memory_values = []
            memory_value = 0
            
            # This still has to be done row by row due to the cumulative nature
            for i in range(len(df)):
                if negative_events.iloc[i]:
                    # New negative event - add to memory
                    memory_value += df[col].iloc[i] * df['tone_negative_max'].iloc[i]
                else:
                    # Decay existing memory
                    memory_value *= (1 - memory_alpha)
                    
                memory_values.append(memory_value)
            
            # Add the complete series to new_features dict
            memory_col = f'{base_name}_memory_effect'
            new_features[memory_col] = pd.Series(memory_values, index=df.index)
            
            # Create interaction between current theme and memory effect
            new_features[f'{base_name}_memory_interaction'] = df[col] * pd.Series(memory_values, index=df.index)
    
    # Add all new features at once to avoid fragmentation
    result_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    
    return result_df

if __name__ == "__main__":
    # This allows the module to be run standalone for testing
    print("Temporal Effects Analysis Module")
    print("Loading aggregated data...")
    
    # Load aggregated data
    aggregated_path = os.path.join(PATHS.get("AGGREGATED_DIR", ""), "aggregated_gkg_15min.csv")
    if not os.path.exists(aggregated_path):
        print(f"Error: Aggregated data not found at {aggregated_path}")
        sys.exit(1)
    
    df = pd.read_csv(aggregated_path)
    
    # Convert time_bucket to datetime
    if 'time_bucket' in df.columns:
        df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    
    # Process the dataframe with all functions
    df = add_exponential_decay_features(df)
    df = create_temporal_theme_windows(df)
    df = add_time_weighted_features(df)
    df = add_temporal_interaction_features(df)
    df = create_memory_decay_features(df)
    
    # Save the results
    output_dir = PATHS.get("FEATURE_ENGINEERING_DIR", "")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "temporal_effects_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Temporal effects features saved to {output_path}")
    print(f"Added {df.shape[1]} total features")