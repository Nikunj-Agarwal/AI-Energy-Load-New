"""
Feature Engineering Package for GDELT Energy Analysis
----------------------------------------------------
This package provides advanced feature engineering tools for GDELT data
to enhance energy load forecasting models.

Modules:
- theme_intensity: Theme presence and importance metrics
- theme_evolution: Tracking how themes change over time
- theme_occurence: Analyzing theme co-occurrence patterns
- temporal_effects: Modeling how theme impacts decay over time
"""

# Package metadata
__version__ = "0.2.0"
__author__ = "AI Energy Load Forecasting Project"

# Import key functions from submodules for easier access
from .theme_intensity import enhance_theme_intensity
from .theme_evolution import (
    add_momentum_features, 
    detect_theme_novelty,
    analyze_theme_persistence,
    detect_theme_seasonality,
    analyze_long_term_trends,
    add_cumulative_features
)
from .theme_occurence import (
    extract_theme_cooccurrence,
    extract_time_based_theme_sequences,
    compute_intertemporal_cooccurrence
)
from .temporal_effects import (
    add_exponential_decay_features,
    create_temporal_theme_windows,
    add_time_weighted_features,
    add_temporal_interaction_features,
    create_memory_decay_features
)

# Define public API
__all__ = [
    'enhance_theme_intensity',
    'add_momentum_features',
    'detect_theme_novelty',
    'analyze_theme_persistence',
    'detect_theme_seasonality',
    'analyze_long_term_trends',
    'add_cumulative_features',
    'extract_theme_cooccurrence',
    'extract_time_based_theme_sequences',
    'compute_intertemporal_cooccurrence',
    'add_exponential_decay_features',
    'create_temporal_theme_windows', 
    'add_time_weighted_features',
    'add_temporal_interaction_features',
    'create_memory_decay_features'
]