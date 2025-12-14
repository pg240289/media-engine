"""
Analytics Engine - Core metrics computation and aggregation

All computations are fully dynamic - no hardcoding.
Works with any dataset following the schema.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from utils.formatters import format_inr, format_number, format_percentage, format_delta, format_ratio


class AnalyticsEngine:
    """
    Core analytics engine for computing marketing metrics.
    All methods are dynamic and work with any compliant dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame

        Args:
            df: DataFrame with columns: date, platform, campaign_name, channel,
                geo, spend, impressions, clicks, conversions, revenue
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all derived metrics for a DataFrame

        Args:
            df: DataFrame to compute metrics for

        Returns:
            Dictionary with all computed metrics
        """
        # Aggregate base metrics
        spend = df['spend'].sum()
        impressions = df['impressions'].sum()
        clicks = df['clicks'].sum()
        conversions = df['conversions'].sum()
        revenue = df['revenue'].sum()

        # Compute derived metrics with safe division
        cpa = spend / conversions if conversions > 0 else 0
        roas = revenue / spend if spend > 0 else 0
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0
        cpc = spend / clicks if clicks > 0 else 0
        cpm = (spend / impressions * 1000) if impressions > 0 else 0

        return {
            'spend': spend,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'revenue': revenue,
            'cpa': cpa,
            'roas': roas,
            'ctr': ctr,
            'cvr': cvr,
            'cpc': cpc,
            'cpm': cpm,
        }

    def get_summary_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get summary metrics for filtered data

        Returns:
            Dictionary with current metrics, previous period metrics, and deltas
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if len(filtered) == 0:
            return self._empty_summary()

        # Current period metrics
        current_metrics = self.compute_metrics(filtered)

        # Previous period (same duration)
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration = (end - start).days

            prev_end = start - pd.Timedelta(days=1)
            prev_start = prev_end - pd.Timedelta(days=duration)

            prev_filtered = self._filter_data(
                prev_start.strftime('%Y-%m-%d'),
                prev_end.strftime('%Y-%m-%d'),
                platforms,
                geos
            )
            previous_metrics = self.compute_metrics(prev_filtered) if len(prev_filtered) > 0 else None
        else:
            previous_metrics = None

        # Compute deltas
        deltas = {}
        if previous_metrics:
            for metric in current_metrics:
                deltas[metric] = format_delta(
                    current_metrics[metric],
                    previous_metrics[metric]
                )

        return {
            'current': current_metrics,
            'previous': previous_metrics,
            'deltas': deltas,
            'period': {
                'start': start_date,
                'end': end_date,
            }
        }

    def get_breakdown_by_dimension(
        self,
        dimension: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        include_totals: bool = True,
    ) -> pd.DataFrame:
        """
        Get metrics broken down by a dimension (platform, ad_type, geo, campaign_name)

        Args:
            dimension: Column to group by
            include_totals: Whether to include a totals row

        Returns:
            DataFrame with metrics per dimension value
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if len(filtered) == 0:
            return pd.DataFrame()

        # Aggregate by dimension
        grouped = filtered.groupby(dimension).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        # Compute derived metrics
        grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        grouped['cvr'] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100
        grouped['cpc'] = grouped['spend'] / grouped['clicks'].replace(0, np.nan)

        # Fill NaN with 0
        grouped = grouped.fillna(0)

        # Compute shares
        total_spend = grouped['spend'].sum()
        total_conversions = grouped['conversions'].sum()
        grouped['spend_share'] = grouped['spend'] / total_spend * 100 if total_spend > 0 else 0
        grouped['conv_share'] = grouped['conversions'] / total_conversions * 100 if total_conversions > 0 else 0

        # Sort by spend descending
        grouped = grouped.sort_values('spend', ascending=False)

        if include_totals:
            totals = pd.DataFrame([{
                dimension: 'Total',
                'spend': grouped['spend'].sum(),
                'impressions': grouped['impressions'].sum(),
                'clicks': grouped['clicks'].sum(),
                'conversions': grouped['conversions'].sum(),
                'revenue': grouped['revenue'].sum(),
                'cpa': grouped['spend'].sum() / grouped['conversions'].sum() if grouped['conversions'].sum() > 0 else 0,
                'roas': grouped['revenue'].sum() / grouped['spend'].sum() if grouped['spend'].sum() > 0 else 0,
                'ctr': grouped['clicks'].sum() / grouped['impressions'].sum() * 100 if grouped['impressions'].sum() > 0 else 0,
                'cvr': grouped['conversions'].sum() / grouped['clicks'].sum() * 100 if grouped['clicks'].sum() > 0 else 0,
                'cpc': grouped['spend'].sum() / grouped['clicks'].sum() if grouped['clicks'].sum() > 0 else 0,
                'spend_share': 100.0,
                'conv_share': 100.0,
            }])
            grouped = pd.concat([grouped, totals], ignore_index=True)

        return grouped

    def get_time_series(
        self,
        metric: str,
        granularity: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        breakdown: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get time series data for a metric

        Args:
            metric: Metric to compute (spend, cpa, roas, ctr, cvr, conversions, etc.)
            granularity: 'daily' or 'weekly'
            breakdown: Optional dimension to break down by

        Returns:
            DataFrame with date and metric columns
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if len(filtered) == 0:
            return pd.DataFrame()

        # Set up grouping
        filtered = filtered.copy()
        if granularity == 'weekly':
            filtered['period'] = filtered['date'].dt.to_period('W').dt.start_time
        else:
            filtered['period'] = filtered['date']

        group_cols = ['period']
        if breakdown:
            group_cols.append(breakdown)

        # Aggregate
        grouped = filtered.groupby(group_cols).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        # Compute derived metrics
        if metric == 'cpa':
            grouped[metric] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        elif metric == 'roas':
            grouped[metric] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        elif metric == 'ctr':
            grouped[metric] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        elif metric == 'cvr':
            grouped[metric] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100
        elif metric == 'cpc':
            grouped[metric] = grouped['spend'] / grouped['clicks'].replace(0, np.nan)
        elif metric in ['spend', 'impressions', 'clicks', 'conversions', 'revenue']:
            pass  # Already computed
        else:
            raise ValueError(f"Unknown metric: {metric}")

        grouped = grouped.fillna(0)

        return grouped

    def get_comparison(
        self,
        dimension: str,
        current_start: str,
        current_end: str,
        previous_start: str,
        previous_end: str,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare metrics across two time periods

        Returns:
            DataFrame with current, previous, and delta columns
        """
        # Current period
        current_df = self._filter_data(current_start, current_end, platforms, geos)
        current_breakdown = self.get_breakdown_by_dimension(
            dimension, current_start, current_end, platforms, geos, include_totals=False
        )

        # Previous period
        previous_df = self._filter_data(previous_start, previous_end, platforms, geos)
        previous_breakdown = self.get_breakdown_by_dimension(
            dimension, previous_start, previous_end, platforms, geos, include_totals=False
        )

        # Merge
        merged = current_breakdown.merge(
            previous_breakdown,
            on=dimension,
            suffixes=('_current', '_previous'),
            how='outer'
        ).fillna(0)

        # Compute deltas for key metrics
        for metric in ['spend', 'conversions', 'cpa', 'roas', 'cvr']:
            curr_col = f'{metric}_current'
            prev_col = f'{metric}_previous'
            if curr_col in merged.columns and prev_col in merged.columns:
                merged[f'{metric}_delta'] = merged[curr_col] - merged[prev_col]
                merged[f'{metric}_delta_pct'] = (
                    (merged[curr_col] - merged[prev_col]) /
                    merged[prev_col].replace(0, np.nan) * 100
                ).fillna(0)

        return merged

    def get_top_performers(
        self,
        dimension: str,
        metric: str,
        n: int = 5,
        ascending: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get top N performers by a metric"""
        breakdown = self.get_breakdown_by_dimension(
            dimension, start_date, end_date, platforms, geos, include_totals=False
        )

        return breakdown.nsmallest(n, metric) if ascending else breakdown.nlargest(n, metric)

    def _filter_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply filters to the DataFrame"""
        filtered = self.df.copy()

        if start_date:
            filtered = filtered[filtered['date'] >= pd.to_datetime(start_date)]

        if end_date:
            filtered = filtered[filtered['date'] <= pd.to_datetime(end_date)]

        if platforms and len(platforms) > 0:
            filtered = filtered[filtered['platform'].isin(platforms)]

        if geos and len(geos) > 0:
            filtered = filtered[filtered['geo'].isin(geos)]

        return filtered

    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure"""
        empty_metrics = {
            'spend': 0, 'impressions': 0, 'clicks': 0, 'conversions': 0,
            'revenue': 0, 'cpa': 0, 'roas': 0, 'ctr': 0, 'cvr': 0, 'cpc': 0, 'cpm': 0
        }
        return {
            'current': empty_metrics,
            'previous': None,
            'deltas': {},
            'period': {'start': None, 'end': None}
        }

    def format_breakdown_for_display(self, breakdown_df: pd.DataFrame, dimension: str) -> List[Dict]:
        """
        Format breakdown DataFrame for UI display

        Returns:
            List of dictionaries with formatted values
        """
        rows = []
        for _, row in breakdown_df.iterrows():
            rows.append({
                dimension: row[dimension],
                'spend': format_inr(row['spend']),
                'spend_raw': row['spend'],
                'conversions': format_number(row['conversions']),
                'conversions_raw': row['conversions'],
                'cpa': format_inr(row['cpa']),
                'cpa_raw': row['cpa'],
                'roas': format_ratio(row['roas']),
                'roas_raw': row['roas'],
                'ctr': format_percentage(row['ctr']),
                'ctr_raw': row['ctr'],
                'cvr': format_percentage(row['cvr']),
                'cvr_raw': row['cvr'],
                'spend_share': format_percentage(row['spend_share']),
                'conv_share': format_percentage(row['conv_share']),
            })
        return rows

    # ==================== HIERARCHY METHODS ====================

    def get_hierarchical_breakdown(
        self,
        levels: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        include_totals: bool = False,
    ) -> pd.DataFrame:
        """
        Get metrics broken down by multiple hierarchy levels

        Args:
            levels: List of columns to group by (e.g., ['platform', 'campaign_name', 'objective'])
            filters: Additional filters as dict of column: values

        Returns:
            DataFrame with metrics per combination of dimension values
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        # Apply additional hierarchy filters
        if filters:
            for col, values in filters.items():
                if values and len(values) > 0 and col in filtered.columns:
                    filtered = filtered[filtered[col].isin(values)]

        if len(filtered) == 0:
            return pd.DataFrame()

        # Aggregate by all levels
        grouped = filtered.groupby(levels).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        # Compute derived metrics
        grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        grouped['cvr'] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100
        grouped['cpc'] = grouped['spend'] / grouped['clicks'].replace(0, np.nan)

        grouped = grouped.fillna(0)

        # Compute shares
        total_spend = grouped['spend'].sum()
        total_conversions = grouped['conversions'].sum()
        grouped['spend_share'] = grouped['spend'] / total_spend * 100 if total_spend > 0 else 0
        grouped['conv_share'] = grouped['conversions'] / total_conversions * 100 if total_conversions > 0 else 0

        # Sort by spend descending
        grouped = grouped.sort_values('spend', ascending=False)

        if include_totals:
            totals_row = {level: 'Total' for level in levels}
            totals_row.update({
                'spend': grouped['spend'].sum(),
                'impressions': grouped['impressions'].sum(),
                'clicks': grouped['clicks'].sum(),
                'conversions': grouped['conversions'].sum(),
                'revenue': grouped['revenue'].sum(),
                'cpa': grouped['spend'].sum() / grouped['conversions'].sum() if grouped['conversions'].sum() > 0 else 0,
                'roas': grouped['revenue'].sum() / grouped['spend'].sum() if grouped['spend'].sum() > 0 else 0,
                'ctr': grouped['clicks'].sum() / grouped['impressions'].sum() * 100 if grouped['impressions'].sum() > 0 else 0,
                'cvr': grouped['conversions'].sum() / grouped['clicks'].sum() * 100 if grouped['clicks'].sum() > 0 else 0,
                'cpc': grouped['spend'].sum() / grouped['clicks'].sum() if grouped['clicks'].sum() > 0 else 0,
                'spend_share': 100.0,
                'conv_share': 100.0,
            })
            grouped = pd.concat([grouped, pd.DataFrame([totals_row])], ignore_index=True)

        return grouped

    def get_campaign_performance(
        self,
        campaign_name: Optional[str] = None,
        campaign_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed performance for a specific campaign

        Returns:
            Dict with campaign metrics, breakdown by objective/ad_type/adset, and trends
        """
        filtered = self._filter_data(start_date, end_date)

        if campaign_name:
            filtered = filtered[filtered['campaign_name'] == campaign_name]
        elif campaign_id:
            filtered = filtered[filtered['campaign_id'] == campaign_id]

        if len(filtered) == 0:
            return {'metrics': self._empty_summary()['current'], 'breakdowns': {}}

        metrics = self.compute_metrics(filtered)

        # Get breakdowns
        breakdowns = {}
        for dimension in ['platform', 'objective', 'ad_type', 'adset_name', 'creative_type']:
            if dimension in filtered.columns:
                dim_grouped = filtered.groupby(dimension).agg({
                    'spend': 'sum',
                    'conversions': 'sum',
                    'revenue': 'sum',
                }).reset_index()
                dim_grouped['cpa'] = dim_grouped['spend'] / dim_grouped['conversions'].replace(0, np.nan)
                dim_grouped['roas'] = dim_grouped['revenue'] / dim_grouped['spend'].replace(0, np.nan)
                dim_grouped = dim_grouped.fillna(0).sort_values('spend', ascending=False)
                breakdowns[dimension] = dim_grouped

        return {
            'metrics': metrics,
            'breakdowns': breakdowns,
            'campaign_name': campaign_name or filtered['campaign_name'].iloc[0] if len(filtered) > 0 else None,
        }

    def get_creative_analysis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze performance by creative type and creative details

        Returns:
            Dict with 'by_type', 'by_creative', 'top_performers' DataFrames
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if filters:
            for col, values in filters.items():
                if values and len(values) > 0 and col in filtered.columns:
                    filtered = filtered[filtered[col].isin(values)]

        if len(filtered) == 0:
            return {'by_type': pd.DataFrame(), 'by_creative': pd.DataFrame(), 'top_performers': pd.DataFrame()}

        # Performance by creative type
        by_type = filtered.groupby('creative_type').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()
        by_type['cpa'] = by_type['spend'] / by_type['conversions'].replace(0, np.nan)
        by_type['roas'] = by_type['revenue'] / by_type['spend'].replace(0, np.nan)
        by_type['ctr'] = by_type['clicks'] / by_type['impressions'].replace(0, np.nan) * 100
        by_type['cvr'] = by_type['conversions'] / by_type['clicks'].replace(0, np.nan) * 100
        by_type = by_type.fillna(0).sort_values('spend', ascending=False)

        # Spend share by type
        total_spend = by_type['spend'].sum()
        by_type['spend_share'] = by_type['spend'] / total_spend * 100 if total_spend > 0 else 0

        # Performance by individual creative
        by_creative = filtered.groupby(['creative_id', 'creative_name', 'creative_type', 'platform']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()
        by_creative['cpa'] = by_creative['spend'] / by_creative['conversions'].replace(0, np.nan)
        by_creative['roas'] = by_creative['revenue'] / by_creative['spend'].replace(0, np.nan)
        by_creative['ctr'] = by_creative['clicks'] / by_creative['impressions'].replace(0, np.nan) * 100
        by_creative['cvr'] = by_creative['conversions'] / by_creative['clicks'].replace(0, np.nan) * 100
        by_creative = by_creative.fillna(0).sort_values('spend', ascending=False)

        # Top performers by ROAS (with minimum spend threshold)
        min_spend = by_creative['spend'].quantile(0.25) if len(by_creative) > 10 else 0
        top_performers = by_creative[by_creative['spend'] >= min_spend].nlargest(10, 'roas')

        return {
            'by_type': by_type,
            'by_creative': by_creative,
            'top_performers': top_performers,
        }

    def get_objective_breakdown(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Performance breakdown by campaign objective

        Returns:
            DataFrame with metrics per objective
        """
        return self.get_breakdown_by_dimension(
            'objective',
            start_date=start_date,
            end_date=end_date,
            platforms=platforms,
            geos=geos,
            include_totals=True,
        )

    def get_ad_type_breakdown(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Performance breakdown by ad type

        Returns:
            DataFrame with metrics per ad type
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if filters:
            for col, values in filters.items():
                if values and len(values) > 0 and col in filtered.columns:
                    filtered = filtered[filtered[col].isin(values)]

        if len(filtered) == 0:
            return pd.DataFrame()

        grouped = filtered.groupby(['platform', 'ad_type']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        grouped['cvr'] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100

        grouped = grouped.fillna(0).sort_values('spend', ascending=False)

        return grouped

    def get_adset_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get adset (targeting) level performance

        Returns:
            DataFrame with top N adsets by spend
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if filters:
            for col, values in filters.items():
                if values and len(values) > 0 and col in filtered.columns:
                    filtered = filtered[filtered[col].isin(values)]

        if len(filtered) == 0:
            return pd.DataFrame()

        grouped = filtered.groupby(['adset_id', 'adset_name', 'targeting_type', 'campaign_name', 'platform']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        grouped['cvr'] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100

        grouped = grouped.fillna(0).sort_values('spend', ascending=False)

        return grouped.head(top_n)

    def get_ad_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get individual ad level performance

        Returns:
            DataFrame with top N ads by spend
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if filters:
            for col, values in filters.items():
                if values and len(values) > 0 and col in filtered.columns:
                    filtered = filtered[filtered[col].isin(values)]

        if len(filtered) == 0:
            return pd.DataFrame()

        grouped = filtered.groupby(['ad_id', 'ad_name', 'creative_type', 'adset_name', 'campaign_name', 'platform']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        grouped['cpa'] = grouped['spend'] / grouped['conversions'].replace(0, np.nan)
        grouped['roas'] = grouped['revenue'] / grouped['spend'].replace(0, np.nan)
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace(0, np.nan) * 100
        grouped['cvr'] = grouped['conversions'] / grouped['clicks'].replace(0, np.nan) * 100

        grouped = grouped.fillna(0).sort_values('spend', ascending=False)

        return grouped.head(top_n)
