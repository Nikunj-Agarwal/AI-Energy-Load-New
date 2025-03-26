import pandas as pd
import numpy as np

def enhance_theme_intensity(df):
    """
    Enhances theme intensity features by looking at
    relative importance rather than just presence.
    
    Args:
        df: DataFrame from aggregation step
        
    Returns:
        DataFrame with enhanced theme intensity features
    """
    # Get all theme columns that measure sum (total count)
    theme_sum_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    if not theme_sum_cols:
        return df
    
    # Calculate total theme mentions in each time period
    df['total_theme_mentions'] = df[theme_sum_cols].sum(axis=1)
    
    # Calculate relative importance of each theme
    for theme_col in theme_sum_cols:
        theme_name = theme_col.replace('_sum', '')
        relative_col = f"{theme_name}_relative"
        df[relative_col] = df[theme_col] / df['total_theme_mentions'].replace(0, 1)
    
    # Calculate theme dominance (max relative importance)
    df['dominant_theme_value'] = df[[col for col in df.columns if col.endswith('_relative')]].max(axis=1)
    
    # Create exponential versions of theme features to capture non-linear effects
    for theme_col in theme_sum_cols:
        theme_name = theme_col.replace('_sum', '')
        max_val = df[theme_col].max() or 1  # Use 1 if max is 0
        df[f"{theme_name}_exp"] = np.exp(df[theme_col] / max_val * 2) - 1
    
    # Add theme intensity thresholds (binary indicators for significant coverage)
    for theme_col in theme_sum_cols:
        theme_name = theme_col.replace('_sum', '')
        # 75th percentile threshold
        threshold = df[theme_col].quantile(0.75)
        df[f"{theme_name}_high"] = (df[theme_col] > threshold).astype(int)
        
        # 90th percentile threshold (very high intensity)
        threshold_v_high = df[theme_col].quantile(0.90)
        df[f"{theme_name}_very_high"] = (df[theme_col] > threshold_v_high).astype(int)
    
    # Create theme intensity volatility metric (how much theme importance changes)
    for theme_col in theme_sum_cols:
        theme_name = theme_col.replace('_sum', '')
        relative_col = f"{theme_name}_relative"
        # Rolling standard deviation of theme importance (volatility)
        if len(df) > 4:  # Only if we have enough data
            df[f"{theme_name}_volatility"] = df[relative_col].rolling(4).std().fillna(0)
    
    # Create ratios between theme categories (energy vs infrastructure, etc.)
    if 'theme_Energy_sum' in theme_sum_cols and 'theme_Infrastructure_sum' in theme_sum_cols:
        df['energy_infra_ratio'] = df['theme_Energy_sum'] / df['theme_Infrastructure_sum'].replace(0, 1)
    
    if 'theme_Energy_sum' in theme_sum_cols and 'theme_Environment_sum' in theme_sum_cols:
        df['energy_environment_ratio'] = df['theme_Energy_sum'] / df['theme_Environment_sum'].replace(0, 1)
    
    return df