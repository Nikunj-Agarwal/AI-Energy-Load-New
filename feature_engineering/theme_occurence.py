import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import itertools
from collections import Counter

# Add parent directory to path to ensure we can import config properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import (
    PATHS, THEME_CATEGORIES, ENERGY_THEMES, 
    OUTPUT_DIR, FEATURE_ENGINEERING
)

def extract_theme_cooccurrence(df, min_occurrence=5, max_combinations=10):
    """
    Extract theme co-occurrence patterns from aggregated data
    
    Args:
        df: DataFrame with aggregated theme data
        min_occurrence: Minimum count to include a combination
        max_combinations: Maximum number of theme combinations to create
        
    Returns:
        DataFrame with added co-occurrence features
    """
    print("Extracting theme co-occurrence patterns...")
    
    # Get theme columns ending with _sum (these contain counts)
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    if not theme_cols:
        # Try for article-level data without _sum suffix
        theme_cols = [col for col in df.columns if col.startswith('theme_') and not '_' in col[6:]]
    
    if not theme_cols:
        print("No theme columns found in data. Skipping co-occurrence extraction.")
        return df
    
    # Create co-occurrence features for highly correlated themes
    theme_corrs = pd.DataFrame()
    for t1, t2 in itertools.combinations(theme_cols, 2):
        # Get theme names without the _sum suffix
        t1_name = t1.replace('_sum', '')
        t2_name = t2.replace('_sum', '')
        
        # Skip if we already have this combination from parsing
        combo_name = f"{t1_name}_{t2_name}_combo"
        if combo_name in df.columns:
            continue
            
        # Calculate correlation between themes
        corr = df[t1].corr(df[t2])
        theme_corrs = pd.concat([theme_corrs, pd.DataFrame({
            'theme1': [t1_name],
            'theme2': [t2_name],
            'correlation': [corr],
            'co_occurrence_count': [(df[t1] > 0) & (df[t2] > 0)].sum()
        })])
    
    # Get top combinations by correlation and occurrence count
    top_combos = theme_corrs[theme_corrs['co_occurrence_count'] >= min_occurrence]
    top_combos = top_combos.sort_values('correlation', ascending=False).head(max_combinations)
    
    # Add co-occurrence features
    for _, row in top_combos.iterrows():
        t1 = f"theme_{row['theme1']}_sum"
        t2 = f"theme_{row['theme2']}_sum"
        combo_name = f"{row['theme1']}_{row['theme2']}_combo"
        
        # Binary co-occurrence (both present)
        df[combo_name] = ((df[t1] > 0) & (df[t2] > 0)).astype(int)
        
        # Strength of co-occurrence (product of normalized values)
        df[f"{combo_name}_strength"] = (
            df[t1] / df[t1].max() * 
            df[t2] / df[t2].max()
        )
    
    # Create energy-specific combinations
    create_energy_theme_combinations(df)
    
    # Calculate association rules between themes
    if theme_cols and len(df) > 0:
        calculate_theme_associations(df)
    
    return df

def create_energy_theme_combinations(df):
    """Create specific combinations with energy themes"""
    energy_theme = 'theme_Energy_sum'
    if energy_theme not in df.columns:
        return
    
    # Important combinations with Energy theme
    key_themes = ['Infrastructure', 'Environment', 'Political', 'Economic']
    
    for theme in key_themes:
        other_theme = f'theme_{theme}_sum'
        if other_theme in df.columns:
            # Create binary indicator
            df[f'Energy_{theme}_combo'] = ((df[energy_theme] > 0) & (df[other_theme] > 0)).astype(int)
            
            # Create interaction term weighted by tone
            if 'tone_tone_mean' in df.columns:
                df[f'Energy_{theme}_impact'] = (
                    df[f'Energy_{theme}_combo'] * 
                    df['tone_tone_mean'].abs()
                )
                
                # Negative tone specifically has more impact
                if 'tone_negative_max' in df.columns:
                    df[f'Energy_{theme}_negative_impact'] = (
                        df[f'Energy_{theme}_combo'] * 
                        df['tone_negative_max']
                    )
    
    # Create "supply disruption" feature
    if 'energy_supply' in df.columns and 'tone_negative_max' in df.columns:
        df['supply_disruption_indicator'] = (
            (df['energy_supply'] > 0) & 
            (df['tone_negative_max'] > 0.5)
        ).astype(int)
    
    # Create "demand spike" feature
    if 'energy_demand' in df.columns and 'tone_volatility' in df.columns:
        df['demand_spike_indicator'] = (
            (df['energy_demand'] > 0) & 
            (df['tone_volatility'] > df['tone_volatility'].quantile(0.75))
        ).astype(int)

def calculate_theme_associations(df):
    """Calculate association rules between themes"""
    # Get theme columns (binary presence indicators)
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Convert to binary (present/absent)
    theme_binary = {col: (df[col] > 0).astype(int) for col in theme_cols}
    
    # Initialize lift and confidence metrics
    for t1 in theme_cols:
        t1_name = t1.replace('_sum', '')
        
        for t2 in theme_cols:
            if t1 == t2:
                continue
                
            t2_name = t2.replace('_sum', '')
            
            # Calculate support, confidence and lift
            support_t1 = theme_binary[t1].mean()
            support_t2 = theme_binary[t2].mean()
            support_t1_t2 = (theme_binary[t1] & theme_binary[t2]).mean()
            
            if support_t1 > 0:
                confidence = support_t1_t2 / support_t1
                if support_t2 > 0:
                    lift = confidence / support_t2
                    
                    # Only add high-lift associations
                    if lift > 1.5:
                        df[f'{t1_name}_implies_{t2_name}'] = lift * (theme_binary[t1] & theme_binary[t2])

def extract_time_based_theme_sequences(df, window_size=4):
    """
    Extract sequences of themes that appear in a time window
    
    Args:
        df: DataFrame with time-ordered theme data
        window_size: Number of consecutive time periods to check
        
    Returns:
        DataFrame with added theme sequence features
    """
    # Ensure we have time-sorted data
    if 'time_bucket' not in df.columns:
        print("No time_bucket column found. Cannot extract time sequences.")
        return df
    
    # Sort by time
    df = df.sort_values('time_bucket')
    
    # Get theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    theme_names = [col.replace('_sum', '') for col in theme_cols]
    
    # Create rolling binary presence
    for theme in theme_names:
        col = f'theme_{theme}_sum'
        presence = (df[col] > 0).astype(int)
        
        # Calculate consecutive appearances
        df[f'{theme}_consecutive'] = presence.groupby(
            (presence != presence.shift(1)).cumsum()
        ).cumcount() + 1
        
        # Create features for themes appearing in consecutive intervals
        df[f'{theme}_streak'] = df[f'{theme}_consecutive'].apply(
            lambda x: min(x, window_size)  # Cap at window_size
        )
        
        # Calculate theme persistence (how long a theme stays relevant)
        for w in range(2, window_size+1):
            df[f'{theme}_present_{w}periods'] = (
                presence.rolling(window=w, min_periods=1).sum() == w
            ).astype(int)
    
    return df

def compute_intertemporal_cooccurrence(df, lag_periods=(1, 2, 3), theme_columns=None):
    """
    Compute co-occurrences of themes across time periods
    
    Args:
        df: DataFrame with time-ordered theme data
        lag_periods: Time lags to check for co-occurrences
        theme_columns: List of theme columns to use, or None for all theme columns
        
    Returns:
        DataFrame with added intertemporal co-occurrence features
    """
    if theme_columns is None:
        theme_columns = [col for col in df.columns if col.startswith('theme_') and col.endswith('_sum')]
    
    # Make sure data is time-ordered
    df = df.sort_values('time_bucket')
    
    # For key themes, check if they're followed by other themes
    key_themes = ['Energy', 'Political', 'Economic', 'Environment']
    key_theme_cols = [f'theme_{t}_sum' for t in key_themes if f'theme_{t}_sum' in df.columns]
    
    for lag in lag_periods:
        for theme_col in key_theme_cols:
            theme_name = theme_col.replace('_sum', '')
            
            # Shift to create lag features
            df[f'{theme_name}_lag{lag}'] = df[theme_col].shift(lag)
            
            # Create features that capture if theme A is followed by theme B in subsequent periods
            for other_col in theme_columns:
                if theme_col == other_col:
                    continue
                    
                other_name = other_col.replace('_sum', '')
                feature_name = f'{theme_name}_followed_by_{other_name}_lag{lag}'
                
                # Theme A in current period and Theme B in future period
                df[feature_name] = ((df[theme_col] > 0) & (df[other_col].shift(-lag) > 0)).astype(int)
    
    return df

if __name__ == "__main__":
    # This allows the module to be run standalone for testing
    print("Theme Co-occurrence Analysis Module")
    print("Loading aggregated data...")
    
    # Load aggregated data
    aggregated_path = os.path.join(PATHS.get("AGGREGATED_DIR", ""), "aggregated_gkg_15min.csv")
    if not os.path.exists(aggregated_path):
        print(f"Error: Aggregated data not found at {aggregated_path}")
        sys.exit(1)
    
    df = pd.read_csv(aggregated_path)
    
    # Process the dataframe with all functions
    df = extract_theme_cooccurrence(df)
    df = extract_time_based_theme_sequences(df)
    df = compute_intertemporal_cooccurrence(df)
    
    # Save the results
    output_dir = PATHS.get("FEATURE_ENGINEERING_DIR", "")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "theme_cooccurrence_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Theme co-occurrence features saved to {output_path}")
    print(f"Added {df.shape[1]} total features")