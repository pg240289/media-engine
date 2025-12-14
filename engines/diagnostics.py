"""
Diagnostics Engine - Performance analysis and root cause identification

All logic is algorithmic - no hardcoded explanations.
Decomposition and attribution are mathematically derived.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .analytics import AnalyticsEngine


class DiagnosticsEngine:
    """
    Diagnostic engine for identifying performance drivers and root causes.
    Uses mathematical decomposition - no hardcoded templates.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame"""
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.analytics = AnalyticsEngine(df)

    def analyze_period_change(
        self,
        current_start: str,
        current_end: str,
        previous_start: str,
        previous_end: str,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of performance change between two periods

        Returns structured analysis including:
        - Overall metrics change
        - CPA decomposition (CPC vs CVR contribution)
        - Platform-level attribution
        - Top drivers
        """
        # Get filtered data for both periods
        current_df = self._filter_data(current_start, current_end, platforms, geos)
        previous_df = self._filter_data(previous_start, previous_end, platforms, geos)

        if len(current_df) == 0 or len(previous_df) == 0:
            return self._empty_analysis()

        # Compute overall metrics
        current_metrics = self.analytics.compute_metrics(current_df)
        previous_metrics = self.analytics.compute_metrics(previous_df)

        # CPA decomposition
        cpa_decomposition = self._decompose_cpa_change(current_metrics, previous_metrics)

        # Platform attribution
        platform_attribution = self._compute_platform_attribution(
            current_df, previous_df, current_metrics, previous_metrics
        )

        # Geo attribution
        geo_attribution = self._compute_dimension_attribution(
            current_df, previous_df, 'geo'
        )

        # Ad Type attribution (replaces channel in new hierarchy)
        ad_type_attribution = self._compute_dimension_attribution(
            current_df, previous_df, 'ad_type'
        )

        # Generate diagnostic summary
        diagnostic_summary = self._generate_diagnostic_summary(
            current_metrics,
            previous_metrics,
            cpa_decomposition,
            platform_attribution
        )

        return {
            'current_metrics': current_metrics,
            'previous_metrics': previous_metrics,
            'metric_changes': self._compute_metric_changes(current_metrics, previous_metrics),
            'cpa_decomposition': cpa_decomposition,
            'platform_attribution': platform_attribution,
            'geo_attribution': geo_attribution,
            'ad_type_attribution': ad_type_attribution,
            'diagnostic_summary': diagnostic_summary,
            'period': {
                'current': {'start': current_start, 'end': current_end},
                'previous': {'start': previous_start, 'end': previous_end},
            }
        }

    def _decompose_cpa_change(
        self,
        current: Dict[str, float],
        previous: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Mathematically decompose CPA change into CPC and CVR contributions

        Formula: CPA = CPC / CVR (where CVR is expressed as decimal)

        Change attribution:
        - ∆CPA from CPC = (CPC_new - CPC_old) / CVR_old
        - ∆CPA from CVR = CPC_old × (1/CVR_new - 1/CVR_old)
        """
        cpc_curr = current['cpc']
        cpc_prev = previous['cpc']
        cvr_curr = current['cvr'] / 100  # Convert to decimal
        cvr_prev = previous['cvr'] / 100

        cpa_curr = current['cpa']
        cpa_prev = previous['cpa']

        # Avoid division by zero
        if cvr_prev == 0 or cvr_curr == 0 or cpa_prev == 0:
            return {
                'total_change': cpa_curr - cpa_prev,
                'total_change_pct': 0,
                'cpc_contribution': 0,
                'cpc_contribution_pct': 0,
                'cvr_contribution': 0,
                'cvr_contribution_pct': 0,
                'primary_driver': 'insufficient_data',
                'cpc_change': 0,
                'cpc_change_pct': 0,
                'cvr_change': 0,
                'cvr_change_pct': 0,
            }

        # Total CPA change
        cpa_change = cpa_curr - cpa_prev
        cpa_change_pct = (cpa_change / cpa_prev) * 100

        # Decomposition using mathematical attribution
        # Contribution from CPC change (holding CVR constant at previous level)
        cpc_contribution = (cpc_curr - cpc_prev) / cvr_prev

        # Contribution from CVR change (holding CPC constant at previous level)
        cvr_contribution = cpc_prev * (1/cvr_curr - 1/cvr_prev)

        # Calculate percentage contribution to total change
        total_contribution = abs(cpc_contribution) + abs(cvr_contribution)
        if total_contribution > 0:
            cpc_contrib_pct = (abs(cpc_contribution) / total_contribution) * 100
            cvr_contrib_pct = (abs(cvr_contribution) / total_contribution) * 100
        else:
            cpc_contrib_pct = 50
            cvr_contrib_pct = 50

        # Determine primary driver
        if abs(cpc_contribution) > abs(cvr_contribution) * 1.5:
            primary_driver = 'cpc'
        elif abs(cvr_contribution) > abs(cpc_contribution) * 1.5:
            primary_driver = 'cvr'
        else:
            primary_driver = 'both'

        # CPC and CVR changes
        cpc_change = cpc_curr - cpc_prev
        cpc_change_pct = (cpc_change / cpc_prev) * 100 if cpc_prev > 0 else 0
        cvr_change = current['cvr'] - previous['cvr']  # In percentage points
        cvr_change_pct = (cvr_change / previous['cvr']) * 100 if previous['cvr'] > 0 else 0

        return {
            'total_change': cpa_change,
            'total_change_pct': cpa_change_pct,
            'cpc_contribution': cpc_contribution,
            'cpc_contribution_pct': cpc_contrib_pct,
            'cvr_contribution': cvr_contribution,
            'cvr_contribution_pct': cvr_contrib_pct,
            'primary_driver': primary_driver,
            'cpc_change': cpc_change,
            'cpc_change_pct': cpc_change_pct,
            'cvr_change': cvr_change,
            'cvr_change_pct': cvr_change_pct,
            'current_cpc': cpc_curr,
            'previous_cpc': cpc_prev,
            'current_cvr': current['cvr'],
            'previous_cvr': previous['cvr'],
        }

    def _compute_platform_attribution(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        current_total: Dict[str, float],
        previous_total: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Compute each platform's contribution to overall CPA change

        Uses weighted attribution based on spend share
        """
        attributions = []

        # Get platform-level metrics
        platforms = set(current_df['platform'].unique()) | set(previous_df['platform'].unique())

        total_cpa_change = current_total['cpa'] - previous_total['cpa']

        for platform in platforms:
            curr_platform = current_df[current_df['platform'] == platform]
            prev_platform = previous_df[previous_df['platform'] == platform]

            curr_metrics = self.analytics.compute_metrics(curr_platform) if len(curr_platform) > 0 else None
            prev_metrics = self.analytics.compute_metrics(prev_platform) if len(prev_platform) > 0 else None

            if curr_metrics is None or prev_metrics is None:
                continue

            # Platform CPA change
            platform_cpa_change = curr_metrics['cpa'] - prev_metrics['cpa']
            platform_cpa_change_pct = (platform_cpa_change / prev_metrics['cpa'] * 100) if prev_metrics['cpa'] > 0 else 0

            # Spend share (average of current and previous)
            curr_spend_share = curr_metrics['spend'] / current_total['spend'] if current_total['spend'] > 0 else 0
            prev_spend_share = prev_metrics['spend'] / previous_total['spend'] if previous_total['spend'] > 0 else 0
            avg_spend_share = (curr_spend_share + prev_spend_share) / 2

            # Weighted contribution to total CPA change
            weighted_contribution = platform_cpa_change * avg_spend_share

            # Percentage of total change attributed to this platform
            contribution_pct = (weighted_contribution / total_cpa_change * 100) if total_cpa_change != 0 else 0

            # Decompose platform's CPA change
            platform_decomposition = self._decompose_cpa_change(curr_metrics, prev_metrics)

            attributions.append({
                'platform': platform,
                'current_cpa': curr_metrics['cpa'],
                'previous_cpa': prev_metrics['cpa'],
                'cpa_change': platform_cpa_change,
                'cpa_change_pct': platform_cpa_change_pct,
                'spend_share': avg_spend_share * 100,
                'weighted_contribution': weighted_contribution,
                'contribution_pct': contribution_pct,
                'current_spend': curr_metrics['spend'],
                'current_conversions': curr_metrics['conversions'],
                'current_cvr': curr_metrics['cvr'],
                'previous_cvr': prev_metrics['cvr'],
                'cvr_change_pct': platform_decomposition['cvr_change_pct'],
                'current_cpc': curr_metrics['cpc'],
                'previous_cpc': prev_metrics['cpc'],
                'cpc_change_pct': platform_decomposition['cpc_change_pct'],
                'decomposition': platform_decomposition,
            })

        # Sort by absolute contribution
        attributions.sort(key=lambda x: abs(x['weighted_contribution']), reverse=True)

        return attributions

    def _compute_dimension_attribution(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        dimension: str,
    ) -> List[Dict[str, Any]]:
        """
        Compute attribution for any dimension (geo, channel, etc.)
        """
        attributions = []

        values = set(current_df[dimension].unique()) | set(previous_df[dimension].unique())

        total_curr = self.analytics.compute_metrics(current_df)
        total_prev = self.analytics.compute_metrics(previous_df)
        total_cpa_change = total_curr['cpa'] - total_prev['cpa']

        for value in values:
            curr_subset = current_df[current_df[dimension] == value]
            prev_subset = previous_df[previous_df[dimension] == value]

            if len(curr_subset) == 0 or len(prev_subset) == 0:
                continue

            curr_metrics = self.analytics.compute_metrics(curr_subset)
            prev_metrics = self.analytics.compute_metrics(prev_subset)

            cpa_change = curr_metrics['cpa'] - prev_metrics['cpa']
            cpa_change_pct = (cpa_change / prev_metrics['cpa'] * 100) if prev_metrics['cpa'] > 0 else 0

            spend_share = (curr_metrics['spend'] / total_curr['spend'] * 100) if total_curr['spend'] > 0 else 0
            weighted_contribution = cpa_change * (spend_share / 100)
            contribution_pct = (weighted_contribution / total_cpa_change * 100) if total_cpa_change != 0 else 0

            attributions.append({
                dimension: value,
                'current_cpa': curr_metrics['cpa'],
                'previous_cpa': prev_metrics['cpa'],
                'cpa_change': cpa_change,
                'cpa_change_pct': cpa_change_pct,
                'spend_share': spend_share,
                'weighted_contribution': weighted_contribution,
                'contribution_pct': contribution_pct,
            })

        attributions.sort(key=lambda x: abs(x['weighted_contribution']), reverse=True)

        return attributions

    def _compute_metric_changes(
        self,
        current: Dict[str, float],
        previous: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compute change metrics for all KPIs"""
        changes = {}
        for metric in current:
            curr_val = current[metric]
            prev_val = previous[metric]
            change = curr_val - prev_val
            change_pct = (change / prev_val * 100) if prev_val != 0 else 0

            changes[metric] = {
                'current': curr_val,
                'previous': prev_val,
                'change': change,
                'change_pct': change_pct,
            }
        return changes

    def _generate_diagnostic_summary(
        self,
        current: Dict[str, float],
        previous: Dict[str, float],
        cpa_decomposition: Dict[str, Any],
        platform_attribution: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate structured diagnostic summary

        This is NOT a hardcoded template - it's constructed from computed values
        """
        # Determine overall performance direction
        cpa_change = cpa_decomposition['total_change']
        cpa_change_pct = cpa_decomposition['total_change_pct']

        if cpa_change > 0:
            performance_direction = 'declining'
            alert_level = 'warning' if cpa_change_pct > 10 else 'info'
        elif cpa_change < 0:
            performance_direction = 'improving'
            alert_level = 'success'
        else:
            performance_direction = 'stable'
            alert_level = 'info'

        # Identify primary driver
        primary_driver = cpa_decomposition['primary_driver']

        # Find top contributing platform
        top_platform = platform_attribution[0] if platform_attribution else None

        # Build structured findings (not templates - computed)
        findings = []

        # Finding 1: Overall CPA change
        findings.append({
            'type': 'cpa_change',
            'metric': 'CPA',
            'direction': 'increased' if cpa_change > 0 else 'decreased',
            'current_value': current['cpa'],
            'previous_value': previous['cpa'],
            'change_absolute': abs(cpa_change),
            'change_pct': abs(cpa_change_pct),
        })

        # Finding 2: Primary driver
        if primary_driver == 'cvr':
            findings.append({
                'type': 'primary_driver',
                'driver': 'CVR',
                'direction': 'declined' if cpa_decomposition['cvr_change'] < 0 else 'improved',
                'current_value': cpa_decomposition['current_cvr'],
                'previous_value': cpa_decomposition['previous_cvr'],
                'change_pct': abs(cpa_decomposition['cvr_change_pct']),
                'contribution_pct': cpa_decomposition['cvr_contribution_pct'],
            })
        elif primary_driver == 'cpc':
            findings.append({
                'type': 'primary_driver',
                'driver': 'CPC',
                'direction': 'increased' if cpa_decomposition['cpc_change'] > 0 else 'decreased',
                'current_value': cpa_decomposition['current_cpc'],
                'previous_value': cpa_decomposition['previous_cpc'],
                'change_pct': abs(cpa_decomposition['cpc_change_pct']),
                'contribution_pct': cpa_decomposition['cpc_contribution_pct'],
            })
        else:
            findings.append({
                'type': 'primary_driver',
                'driver': 'both CPC and CVR',
                'cpc_change_pct': cpa_decomposition['cpc_change_pct'],
                'cvr_change_pct': cpa_decomposition['cvr_change_pct'],
            })

        # Finding 3: Top platform contributor
        if top_platform and abs(top_platform['contribution_pct']) > 20:
            findings.append({
                'type': 'platform_attribution',
                'platform': top_platform['platform'],
                'contribution_pct': abs(top_platform['contribution_pct']),
                'platform_cpa_change_pct': top_platform['cpa_change_pct'],
                'platform_cvr_change_pct': top_platform['cvr_change_pct'],
                'spend_share': top_platform['spend_share'],
            })

        return {
            'performance_direction': performance_direction,
            'alert_level': alert_level,
            'findings': findings,
            'cpa_decomposition': cpa_decomposition,
        }

    def identify_anomalies(
        self,
        start_date: str,
        end_date: str,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        threshold_std: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Identify anomalous performance using statistical thresholds

        Uses z-score to identify values outside normal range
        """
        filtered = self._filter_data(start_date, end_date, platforms, geos)

        if len(filtered) < 7:  # Need minimum data
            return []

        anomalies = []

        # Group by date and compute daily metrics
        daily = filtered.groupby('date').agg({
            'spend': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'impressions': 'sum',
            'revenue': 'sum',
        }).reset_index()

        daily['cpa'] = daily['spend'] / daily['conversions'].replace(0, np.nan)
        daily['cvr'] = daily['conversions'] / daily['clicks'].replace(0, np.nan) * 100
        daily['cpc'] = daily['spend'] / daily['clicks'].replace(0, np.nan)

        # Check each metric for anomalies
        for metric in ['cpa', 'cvr', 'cpc', 'spend', 'conversions']:
            if metric not in daily.columns:
                continue

            values = daily[metric].dropna()
            if len(values) < 7:
                continue

            mean = values.mean()
            std = values.std()

            if std == 0:
                continue

            # Find anomalies
            for idx, row in daily.iterrows():
                if pd.isna(row[metric]):
                    continue

                z_score = (row[metric] - mean) / std

                if abs(z_score) > threshold_std:
                    anomalies.append({
                        'date': row['date'],
                        'metric': metric,
                        'value': row[metric],
                        'mean': mean,
                        'std': std,
                        'z_score': z_score,
                        'direction': 'high' if z_score > 0 else 'low',
                    })

        return anomalies

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

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'current_metrics': {},
            'previous_metrics': {},
            'metric_changes': {},
            'cpa_decomposition': {},
            'platform_attribution': [],
            'geo_attribution': [],
            'ad_type_attribution': [],
            'diagnostic_summary': {
                'performance_direction': 'unknown',
                'alert_level': 'info',
                'findings': [],
            },
            'period': {}
        }
