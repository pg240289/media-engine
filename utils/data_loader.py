"""
Data loading and DuckDB connection management
"""

import duckdb
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "campaign_performance.csv"

# Try to use Streamlit caching if available
try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False


def get_connection():
    """Get DuckDB connection"""
    conn = duckdb.connect(':memory:')
    return conn


def _load_data_impl() -> pd.DataFrame:
    """
    Load campaign performance data from CSV (implementation)

    Returns:
        DataFrame with all campaign data
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date')

    return df


def load_data() -> pd.DataFrame:
    """
    Load campaign performance data from CSV.
    Uses Streamlit caching when running in Streamlit context.

    Returns:
        DataFrame with all campaign data
    """
    if _has_streamlit:
        try:
            # Use cached version in Streamlit context
            @st.cache_data(ttl=300)
            def _cached_load():
                return _load_data_impl()
            return _cached_load()
        except Exception:
            # Fall back to non-cached if caching fails
            return _load_data_impl()
    else:
        return _load_data_impl()


def load_data_to_duckdb(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Load DataFrame into DuckDB for SQL queries"""
    conn.register('campaign_performance', df)


def get_date_range(df: pd.DataFrame) -> tuple:
    """Get min and max dates from data"""
    return df['date'].min(), df['date'].max()


def get_unique_values(df: pd.DataFrame, column: str) -> list:
    """Get unique values for a column"""
    return sorted(df[column].unique().tolist())


def filter_data(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
    platforms: list = None,
    geos: list = None,
    campaigns: list = None,
    objectives: list = None,
    ad_types: list = None,
    adsets: list = None,
    ads: list = None,
    creative_types: list = None,
) -> pd.DataFrame:
    """
    Filter DataFrame based on criteria

    All filters are optional - if None, no filtering on that dimension

    Hierarchy: Platform → Campaign → Objective → Ad Type → Adset → Ad → Creative
    """
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[filtered['date'] >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[filtered['date'] <= pd.to_datetime(end_date)]

    if platforms is not None and len(platforms) > 0:
        filtered = filtered[filtered['platform'].isin(platforms)]

    if geos is not None and len(geos) > 0:
        filtered = filtered[filtered['geo'].isin(geos)]

    if campaigns is not None and len(campaigns) > 0:
        filtered = filtered[filtered['campaign_name'].isin(campaigns)]

    # New hierarchy filters
    if objectives is not None and len(objectives) > 0:
        filtered = filtered[filtered['objective'].isin(objectives)]

    if ad_types is not None and len(ad_types) > 0:
        filtered = filtered[filtered['ad_type'].isin(ad_types)]

    if adsets is not None and len(adsets) > 0:
        filtered = filtered[filtered['adset_name'].isin(adsets)]

    if ads is not None and len(ads) > 0:
        filtered = filtered[filtered['ad_name'].isin(ads)]

    if creative_types is not None and len(creative_types) > 0:
        filtered = filtered[filtered['creative_type'].isin(creative_types)]

    return filtered


def get_hierarchy_options(df: pd.DataFrame, level: str, parent_filters: dict = None) -> list:
    """
    Get available options for a hierarchy level based on parent filters

    Args:
        df: Source DataFrame
        level: Hierarchy level to get options for
        parent_filters: Dict of parent level filters already applied

    Returns:
        List of unique values available at this level
    """
    filtered = df.copy()

    if parent_filters:
        for col, values in parent_filters.items():
            if values is not None and len(values) > 0:
                filtered = filtered[filtered[col].isin(values)]

    if level in filtered.columns:
        return sorted(filtered[level].unique().tolist())
    return []


def get_hierarchy_data(df: pd.DataFrame, levels: list, metrics_agg: dict = None) -> pd.DataFrame:
    """
    Get aggregated data at specified hierarchy levels

    Args:
        df: Source DataFrame
        levels: List of columns to group by
        metrics_agg: Dict of metric columns and their aggregation functions

    Returns:
        DataFrame grouped by specified levels with aggregated metrics
    """
    if metrics_agg is None:
        metrics_agg = {
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }

    grouped = df.groupby(levels).agg(metrics_agg).reset_index()

    # Compute derived metrics
    grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, float('nan'))
    grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, float('nan'))
    grouped['ctr'] = (grouped['clicks'] / grouped['impressions'].replace(0, float('nan'))) * 100
    grouped['cvr'] = (grouped['conversions'] / grouped['clicks'].replace(0, float('nan'))) * 100
    grouped['cpc'] = grouped['spend'] / grouped['clicks'].replace(0, float('nan'))

    # Fill NaN with 0
    grouped = grouped.fillna(0)

    return grouped


def get_drill_down_data(
    df: pd.DataFrame,
    parent_level: str,
    parent_value: str,
    child_level: str
) -> pd.DataFrame:
    """
    Get breakdown data for drilling down from parent to child level

    Args:
        df: Source DataFrame
        parent_level: Column name of parent level
        parent_value: Value of parent to drill into
        child_level: Column name of child level to show

    Returns:
        DataFrame with child level breakdown
    """
    filtered = df[df[parent_level] == parent_value]
    return get_hierarchy_data(filtered, [child_level])


def get_week_data(df: pd.DataFrame, week_offset: int = 0) -> pd.DataFrame:
    """
    Get data for a specific week

    Args:
        df: Source DataFrame
        week_offset: 0 = current/latest week, -1 = previous week, etc.
    """
    max_date = df['date'].max()

    # Find the Monday of the week containing max_date
    days_since_monday = max_date.weekday()
    current_week_start = max_date - pd.Timedelta(days=days_since_monday)

    # Apply offset (each week is 7 days)
    target_week_start = current_week_start + pd.Timedelta(weeks=week_offset)
    target_week_end = target_week_start + pd.Timedelta(days=6)

    return df[(df['date'] >= target_week_start) & (df['date'] <= target_week_end)]


def get_period_comparison_data(df: pd.DataFrame, current_days: int = 7) -> tuple:
    """
    Get current period and previous period data for comparison

    Returns:
        (current_period_df, previous_period_df)
    """
    max_date = df['date'].max()

    current_start = max_date - pd.Timedelta(days=current_days - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=current_days - 1)

    current_df = df[(df['date'] >= current_start) & (df['date'] <= max_date)]
    previous_df = df[(df['date'] >= previous_start) & (df['date'] <= previous_end)]

    return current_df, previous_df
