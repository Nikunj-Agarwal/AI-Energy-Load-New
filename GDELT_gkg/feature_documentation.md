# GDELT GKG Feature Documentation

## Basic Count & Volume Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `article_count` | Total number of articles in 15-min interval | Count of GKGRECORDID | Measures overall news volume |
| `article_count_change` | Change from previous interval | Current count - previous count | Indicates news activity acceleration |
| `article_volume_spike` | Flag for unusual activity | 1 if count > 1.5x rolling 3hr average | Identifies news bursts |
| `prev_article_count` | Previous interval's count | Shifted article_count | Used for change calculation |

## Theme Presence Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `theme_Energy_sum` | Energy theme mentions | Sum of all Energy theme flags | Measures energy news coverage |
| `theme_Energy_max` | Energy theme presence | Max value (0 or 1) | Binary indicator of any energy news |
| `theme_Energy_mean` | Energy theme avg presence | Mean of theme flags (0-1) | Proportion of energy-themed articles |
| `theme_Environment_sum` | Environment mentions | Sum of theme flags | Weather events, climate coverage |
| `theme_Infrastructure_sum` | Infrastructure mentions | Sum of theme flags | Power grid, road, facility news |
| `theme_Social_sum` | Social events mentions | Sum of theme flags | Gatherings, festivals, protests |
| `theme_Health_sum` | Health topic mentions | Sum of theme flags | Health emergencies, epidemics |
| `theme_Political_sum` | Political mentions | Sum of theme flags | Political events, decisions |
| `theme_Economic_sum` | Economic mentions | Sum of theme flags | Financial, market, economic news |

## Tone & Sentiment Metrics

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `tone_tone_mean` | Average article tone | Mean of tone values | Overall sentiment (-100 to +100) |
| `tone_tone_min` | Most negative tone | Min of tone values | Worst news sentiment |
| `tone_tone_max` | Most positive tone | Max of tone values | Best news sentiment |
| `tone_volatility` | Tone range | tone_max - tone_min | Sentiment polarization |
| `tone_negative_max` | Max negative score | Max of negative values | Strength of negative sentiment |
| `tone_positive_max` | Max positive score | Max of positive values | Strength of positive sentiment |
| `tone_polarity_mean` | Avg opinion polarity | Mean of polarity values | How one-sided sentiment is |
| `tone_activity_mean` | Avg activity score | Mean of activity values | Active vs. passive language |

## Entity & Amount Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `entity_count_sum` | Total entities mentioned | Sum of entity counts | Scale of news coverage |
| `entity_variety_max` | Max entity variety | Max of entity_variety | Diversity of entities in news |
| `max_amount_max` | Largest amount mentioned | Max of max_amount | Scale of numeric values |

## Composite Indicators (Feature Engineering)

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `energy_crisis_indicator` | Energy crisis severity | theme_Energy_sum * tone_negative_max | Power outage/shortage intensity |
| `weather_alert_indicator` | Weather emergency level | theme_Environment_sum * abs(tone_tone_min) | Severity of weather events |
| `social_event_indicator` | Social gathering scale | theme_Social_sum * article_count / 100 | Size of social events/crowds |
| `infrastructure_stress` | Infrastructure problems | theme_Infrastructure_sum * tone_negative_max | Grid/road/facility issues |
| `political_crisis_indicator` | Political crisis level | theme_Political_sum * tone_negative_max | Political instability |
| `economic_impact_indicator` | Economic disruption | theme_Economic_sum * tone_volatility | Market/economic instability |

## Time Context Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `hour` | Hour of day (0-23) | time_bucket.hour | Time of day effects |
| `day_of_week` | Day of week (0-6) | time_bucket.dayofweek | Weekly patterns |
| `is_weekend` | Weekend indicator | 1 if day_of_week >= 5 | Weekend vs. weekday |
| `is_business_hours` | Business hours | 1 if hour 9-17 & weekday | Work hour patterns |
| `month` | Month (1-12) | time_bucket.month | Monthly seasonality |
| `day` | Day of month (1-31) | time_bucket.day | Monthly patterns |
| `hour_sin`, `hour_cos` | Cyclical hour encoding | sin/cos transforms | Circular time patterns |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical day encoding | sin/cos transforms | Circular weekly patterns |