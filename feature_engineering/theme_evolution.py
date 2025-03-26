import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from datetime import datetime, timedelta

# Add parent directory to path to ensure we can import config properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import (
    PATHS, THEME_CATEGORIES, ENERGY_THEMES, 
    OUTPUT_DIR, FEATURE_ENGINEERING
)

def add_momentum_features(df):
    """
    Calculate momentum (rate of change) for theme presence over time
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with added momentum features
    """
    print("Adding theme momentum features...")
    
    # Ensure we have time-sorted data
    if 'time_bucket' not in df.columns:
        print("No time_bucket column found. Cannot calculate momentum.")
        return df
    
    # Sort by time
    df = df.sort_values('time_bucket')
    
    # Get theme columns that represent counts (_sum)
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Calculate momentum (1st derivative) for each theme
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        # Simple momentum (difference from previous period)
        df[f'{base_name}_momentum'] = df[col].diff().fillna(0)
        
        # Normalized momentum (percent change)
        df[f'{base_name}_pct_change'] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        
        # Acceleration (2nd derivative) - change in momentum
        df[f'{base_name}_acceleration'] = df[f'{base_name}_momentum'].diff().fillna(0)
        
        # Smoothed momentum (reduce noise)
        df[f'{base_name}_momentum_smooth'] = df[col].rolling(window=4).mean().diff().fillna(0)
        
        # Create binary indicators for significant changes
        momentum_std = df[f'{base_name}_momentum'].std()
        if momentum_std > 0:
            # Significant increase (> 1 std dev)
            df[f'{base_name}_sig_increase'] = (df[f'{base_name}_momentum'] > momentum_std).astype(int)
            
            # Significant decrease (< -1 std dev)
            df[f'{base_name}_sig_decrease'] = (df[f'{base_name}_momentum'] < -momentum_std).astype(int)
    
    return df

def detect_theme_novelty(df):
    """
    Detect the emergence of new themes or theme spikes
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with added novelty features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # For each theme, determine first appearance and novelty
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        # Create binary presence indicator
        presence = (df[col] > 0).astype(int)
        
        # Check if this is the first time a theme appears (novelty)
        # Use cumsum - first time it's 1 is the first appearance
        df[f'{base_name}_first_appearance'] = (
            (presence == 1) & (presence.shift(1).fillna(0) == 0)
        ).astype(int)
        
        # Calculate how many periods since theme was last mentioned
        absence_streaks = presence.groupby((presence != presence.shift(1)).cumsum())
        periods_since_last = absence_streaks.cumcount()
        
        # Only calculate for periods where theme is present
        df[f'{base_name}_periods_since_last'] = np.where(
            presence == 1,
            periods_since_last, 
            -1  # -1 means not present in this period
        )
        
        # Calculate novelty score: higher if theme appears after long absence
        df[f'{base_name}_novelty_score'] = np.where(
            df[f'{base_name}_periods_since_last'] > 0,
            np.log1p(df[f'{base_name}_periods_since_last']),
            0
        )
    
    return df

def analyze_theme_persistence(df, window_sizes=[4, 12, 24]):
    """
    Analyze how persistently themes appear over different time windows
    
    Args:
        df: DataFrame with time-ordered theme data
        window_sizes: List of window sizes to analyze (number of periods)
        
    Returns:
        DataFrame with added persistence features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # For each theme, calculate persistence over different windows
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        presence = (df[col] > 0).astype(int)
        
        for window in window_sizes:
            # Calculate percentage of periods where theme was present in window
            if len(df) >= window:
                df[f'{base_name}_persistence_{window}'] = presence.rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Calculate consistency (low standard deviation in theme counts)
                df[f'{base_name}_consistency_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                
                # Normalize consistency by the mean to get coefficient of variation
                window_mean = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{base_name}_volatility_{window}'] = df[f'{base_name}_consistency_{window}'] / window_mean.replace(0, 1)
    
    return df

def detect_theme_seasonality(df):
    """
    Detect daily and weekly patterns in theme occurrences
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with added seasonality features
    """
    # Check if we have proper datetime index
    if 'time_bucket' not in df.columns:
        return df
    
    # Ensure time_bucket is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_bucket']):
        df['time_bucket'] = pd.to_datetime(df['time_bucket'])
    
    # Extract time components
    df['hour'] = df['time_bucket'].dt.hour
    df['day_of_week'] = df['time_bucket'].dt.dayofweek
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Calculate hourly and daily averages for each theme
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        # Hourly patterns
        hourly_avg = df.groupby('hour')[col].mean()
        for hour, avg in hourly_avg.items():
            df.loc[df['hour'] == hour, f'{base_name}_hourly_avg'] = avg
        
        # Daily patterns
        daily_avg = df.groupby('day_of_week')[col].mean()
        for day, avg in daily_avg.items():
            df.loc[df['day_of_week'] == day, f'{base_name}_daily_avg'] = avg
        
        # Calculate deviation from typical patterns
        df[f'{base_name}_hour_deviation'] = df[col] - df[f'{base_name}_hourly_avg']
        df[f'{base_name}_day_deviation'] = df[col] - df[f'{base_name}_daily_avg']
        
        # Significant deviations (binary indicators)
        hour_std = df[f'{base_name}_hour_deviation'].std()
        if hour_std > 0:
            df[f'{base_name}_unusual_for_hour'] = (
                df[f'{base_name}_hour_deviation'].abs() > 2 * hour_std
            ).astype(int)
        
        day_std = df[f'{base_name}_day_deviation'].std()
        if day_std > 0:
            df[f'{base_name}_unusual_for_day'] = (
                df[f'{base_name}_day_deviation'].abs() > 2 * day_std
            ).astype(int)
    
    return df

def analyze_long_term_trends(df, trend_periods=[24, 72, 168]):
    """
    Analyze long-term trends in theme occurrence
    
    Args:
        df: DataFrame with time-ordered theme data
        trend_periods: List of periods for trend analysis
        
    Returns:
        DataFrame with added trend features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Calculate trend metrics for each theme
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        for period in trend_periods:
            if len(df) >= period:
                # Moving average trend
                df[f'{base_name}_trend_{period}'] = df[col].rolling(window=period).mean()
                
                # Linear regression slope over the period
                def rolling_slope(series, window):
                    result = [np.nan] * (window - 1)
                    for i in range(window - 1, len(series)):
                        x = np.arange(window)
                        y = series.iloc[i - window + 1:i + 1].values
                        slope, _, _, _, _ = stats.linregress(x, y)
                        result.append(slope)
                    return result
                
                df[f'{base_name}_slope_{period}'] = rolling_slope(df[col], period)
                
                # Trend strength (ratio of trend to actual)
                df[f'{base_name}_trend_strength_{period}'] = df[f'{base_name}_trend_{period}'] / df[col].replace(0, 1)
                
                # Trend acceleration/deceleration
                df[f'{base_name}_trend_accel_{period}'] = df[f'{base_name}_slope_{period}'].diff().fillna(0)
    
    return df

def add_cumulative_features(df):
    """
    Add cumulative features that track total theme mentions over time
    
    Args:
        df: DataFrame with time-ordered theme data
        
    Returns:
        DataFrame with added cumulative features
    """
    # Ensure we have time-sorted data
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Calculate cumulative counts for each theme
    for col in theme_cols:
        base_name = col.replace('_sum', '')
        
        # Cumulative sum of theme mentions
        df[f'{base_name}_cumulative'] = df[col].cumsum()
        
        # Daily cumulative (resets each day)
        if 'time_bucket' in df.columns:
            df['date'] = df['time_bucket'].dt.date
            daily_group = df.groupby('date')[col].cumsum()
            df[f'{base_name}_daily_cumulative'] = daily_group.values
            df.drop('date', axis=1, inplace=True)
        
        # Exponentially weighted cumulative (more weight to recent)
        df[f'{base_name}_exp_weighted'] = df[col].ewm(span=24).mean()
    
    return df

if __name__ == "__main__":
    # This allows the module to be run standalone for testing
    print("Theme Evolution Analysis Module")
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
    df = add_momentum_features(df)
    df = detect_theme_novelty(df)
    df = analyze_theme_persistence(df)
    df = detect_theme_seasonality(df)
    df = analyze_long_term_trends(df)
    df = add_cumulative_features(df)
    
    # Save the results
    output_dir = PATHS.get("FEATURE_ENGINEERING_DIR", "")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "theme_evolution_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Theme evolution features saved to {output_path}")
    print(f"Added {df.shape[1]} total features")