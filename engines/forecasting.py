"""
Forecasting Engine - Trend projection and scenario simulation

Uses statistical methods - no hardcoded predictions.
All forecasts are derived from actual data patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from scipy import stats
from .analytics import AnalyticsEngine


class ForecastingEngine:
    """
    Forecasting engine for trend projection and budget scenarios.
    All predictions are mathematically derived from data.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame"""
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.analytics = AnalyticsEngine(df)

    def project_trend(
        self,
        metric: str,
        periods_ahead: int = 7,
        lookback_days: int = 21,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
        method: str = 'linear',
    ) -> Dict[str, Any]:
        """
        Project a metric forward using trend analysis

        Args:
            metric: Metric to project (cpa, conversions, spend, roas, etc.)
            periods_ahead: Number of days to project
            lookback_days: Days of historical data to use
            method: 'linear' for linear regression, 'rolling' for rolling average

        Returns:
            Dictionary with projections and confidence intervals
        """
        filtered = self._filter_data(None, None, platforms, geos)

        if len(filtered) < 7:
            return self._empty_projection()

        # Get time series for the metric
        max_date = filtered['date'].max()
        start_date = max_date - pd.Timedelta(days=lookback_days)

        ts = self.analytics.get_time_series(
            metric=metric,
            granularity='daily',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=max_date.strftime('%Y-%m-%d'),
            platforms=platforms,
            geos=geos,
        )

        if len(ts) < 7:
            return self._empty_projection()

        # Prepare data
        ts = ts.sort_values('period')
        ts['day_num'] = range(len(ts))
        values = ts[metric].values
        days = ts['day_num'].values

        if method == 'linear':
            projection = self._linear_projection(days, values, periods_ahead)
        else:
            projection = self._rolling_projection(values, periods_ahead)

        # Generate future dates
        future_dates = [
            max_date + pd.Timedelta(days=i+1)
            for i in range(periods_ahead)
        ]

        return {
            'metric': metric,
            'method': method,
            'historical': {
                'dates': ts['period'].tolist(),
                'values': values.tolist(),
            },
            'projection': {
                'dates': future_dates,
                'values': projection['forecast'],
                'lower_bound': projection['lower'],
                'upper_bound': projection['upper'],
            },
            'trend': {
                'direction': 'increasing' if projection['slope'] > 0 else 'decreasing',
                'slope': projection['slope'],
                'slope_pct_per_day': projection['slope_pct'],
                'r_squared': projection.get('r_squared', None),
            },
            'current_value': float(values[-1]),
            'projected_end_value': float(projection['forecast'][-1]),
            'projected_change_pct': (
                (projection['forecast'][-1] - values[-1]) / values[-1] * 100
                if values[-1] != 0 else 0
            ),
        }

    def _linear_projection(
        self,
        days: np.ndarray,
        values: np.ndarray,
        periods_ahead: int,
    ) -> Dict[str, Any]:
        """Linear regression projection"""
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)

        # Project forward
        future_days = np.array([days[-1] + i + 1 for i in range(periods_ahead)])
        forecast = intercept + slope * future_days

        # Compute prediction intervals (approximate)
        n = len(days)
        mean_x = np.mean(days)
        se = std_err * np.sqrt(1 + 1/n + (future_days - mean_x)**2 / np.sum((days - mean_x)**2))
        margin = 1.96 * se * np.std(values)  # 95% CI

        # Ensure non-negative for metrics that can't be negative
        forecast = np.maximum(forecast, 0)
        lower = np.maximum(forecast - margin, 0)
        upper = forecast + margin

        # Slope as percentage of mean
        mean_value = np.mean(values)
        slope_pct = (slope / mean_value * 100) if mean_value != 0 else 0

        return {
            'forecast': forecast.tolist(),
            'lower': lower.tolist(),
            'upper': upper.tolist(),
            'slope': slope,
            'slope_pct': slope_pct,
            'r_squared': r_value ** 2,
        }

    def _rolling_projection(
        self,
        values: np.ndarray,
        periods_ahead: int,
        window: int = 7,
    ) -> Dict[str, Any]:
        """Rolling average projection"""
        # Use last 'window' days average
        recent_mean = np.mean(values[-window:])
        recent_std = np.std(values[-window:])

        # Flat projection with confidence band
        forecast = np.full(periods_ahead, recent_mean)
        margin = 1.96 * recent_std
        lower = np.maximum(forecast - margin, 0)
        upper = forecast + margin

        # Calculate trend from recent data
        if len(values) >= window:
            slope = (np.mean(values[-window//2:]) - np.mean(values[:window//2])) / (window // 2)
        else:
            slope = 0

        mean_value = np.mean(values)
        slope_pct = (slope / mean_value * 100) if mean_value != 0 else 0

        return {
            'forecast': forecast.tolist(),
            'lower': lower.tolist(),
            'upper': upper.tolist(),
            'slope': slope,
            'slope_pct': slope_pct,
        }

    def simulate_budget_scenario(
        self,
        scenario_type: str,
        source_platform: Optional[str] = None,
        target_platform: Optional[str] = None,
        change_pct: float = 10.0,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate budget change scenarios

        Scenario types:
        - 'increase_all': Increase total budget by change_pct
        - 'decrease_all': Decrease total budget by change_pct
        - 'shift': Shift budget from source_platform to target_platform

        Returns projected impact on conversions, CPA, ROAS
        """
        # Get recent performance (last 14 days) to establish efficiency baselines
        max_date = self.df['date'].max()
        start_date = max_date - pd.Timedelta(days=13)

        filtered = self._filter_data(
            start_date.strftime('%Y-%m-%d'),
            max_date.strftime('%Y-%m-%d'),
            platforms,
            geos
        )

        if len(filtered) == 0:
            return self._empty_scenario()

        # Compute current platform-level metrics
        platform_metrics = {}
        for platform in filtered['platform'].unique():
            platform_data = filtered[filtered['platform'] == platform]
            metrics = self.analytics.compute_metrics(platform_data)
            platform_metrics[platform] = {
                'spend': metrics['spend'],
                'conversions': metrics['conversions'],
                'revenue': metrics['revenue'],
                'efficiency': metrics['conversions'] / metrics['spend'] if metrics['spend'] > 0 else 0,
                'revenue_efficiency': metrics['revenue'] / metrics['spend'] if metrics['spend'] > 0 else 0,
                'cpa': metrics['cpa'],
                'roas': metrics['roas'],
            }

        # Current totals
        current_total = self.analytics.compute_metrics(filtered)

        # Apply scenario
        if scenario_type == 'increase_all':
            projected = self._simulate_uniform_change(
                platform_metrics, current_total, change_pct / 100
            )
        elif scenario_type == 'decrease_all':
            projected = self._simulate_uniform_change(
                platform_metrics, current_total, -change_pct / 100
            )
        elif scenario_type == 'shift' and source_platform and target_platform:
            projected = self._simulate_budget_shift(
                platform_metrics, current_total,
                source_platform, target_platform, change_pct / 100
            )
        else:
            return self._empty_scenario()

        return {
            'scenario_type': scenario_type,
            'change_pct': change_pct,
            'source_platform': source_platform,
            'target_platform': target_platform,
            'current': {
                'spend': current_total['spend'],
                'conversions': current_total['conversions'],
                'revenue': current_total['revenue'],
                'cpa': current_total['cpa'],
                'roas': current_total['roas'],
            },
            'projected': projected,
            'impact': {
                'spend_change': projected['spend'] - current_total['spend'],
                'spend_change_pct': (projected['spend'] - current_total['spend']) / current_total['spend'] * 100 if current_total['spend'] > 0 else 0,
                'conversions_change': projected['conversions'] - current_total['conversions'],
                'conversions_change_pct': (projected['conversions'] - current_total['conversions']) / current_total['conversions'] * 100 if current_total['conversions'] > 0 else 0,
                'cpa_change': projected['cpa'] - current_total['cpa'],
                'cpa_change_pct': (projected['cpa'] - current_total['cpa']) / current_total['cpa'] * 100 if current_total['cpa'] > 0 else 0,
                'roas_change': projected['roas'] - current_total['roas'],
                'roas_change_pct': (projected['roas'] - current_total['roas']) / current_total['roas'] * 100 if current_total['roas'] > 0 else 0,
            },
            'platform_breakdown': platform_metrics,
        }

    def _simulate_uniform_change(
        self,
        platform_metrics: Dict,
        current_total: Dict,
        change_ratio: float,
    ) -> Dict[str, float]:
        """Simulate uniform budget increase/decrease"""
        new_spend = current_total['spend'] * (1 + change_ratio)

        # Assume efficiency degrades slightly with increased spend (diminishing returns)
        # and improves slightly with decreased spend
        efficiency_modifier = 1 - (change_ratio * 0.1)  # 10% of change affects efficiency

        new_conversions = 0
        new_revenue = 0

        for platform, metrics in platform_metrics.items():
            platform_new_spend = metrics['spend'] * (1 + change_ratio)
            platform_new_conv = platform_new_spend * metrics['efficiency'] * efficiency_modifier
            platform_new_rev = platform_new_spend * metrics['revenue_efficiency'] * efficiency_modifier
            new_conversions += platform_new_conv
            new_revenue += platform_new_rev

        new_cpa = new_spend / new_conversions if new_conversions > 0 else 0
        new_roas = new_revenue / new_spend if new_spend > 0 else 0

        return {
            'spend': new_spend,
            'conversions': new_conversions,
            'revenue': new_revenue,
            'cpa': new_cpa,
            'roas': new_roas,
        }

    def _simulate_budget_shift(
        self,
        platform_metrics: Dict,
        current_total: Dict,
        source: str,
        target: str,
        shift_ratio: float,
    ) -> Dict[str, float]:
        """Simulate budget shift between platforms"""
        if source not in platform_metrics or target not in platform_metrics:
            return {
                'spend': current_total['spend'],
                'conversions': current_total['conversions'],
                'revenue': current_total['revenue'],
                'cpa': current_total['cpa'],
                'roas': current_total['roas'],
            }

        shift_amount = platform_metrics[source]['spend'] * shift_ratio

        new_conversions = 0
        new_revenue = 0

        for platform, metrics in platform_metrics.items():
            if platform == source:
                new_spend = metrics['spend'] - shift_amount
            elif platform == target:
                new_spend = metrics['spend'] + shift_amount
            else:
                new_spend = metrics['spend']

            # Efficiency adjustment for target (diminishing returns with more spend)
            if platform == target:
                efficiency_adj = 0.95  # Slight efficiency loss
            elif platform == source:
                efficiency_adj = 1.02  # Slight efficiency gain from reduced spend
            else:
                efficiency_adj = 1.0

            new_conv = new_spend * metrics['efficiency'] * efficiency_adj
            new_rev = new_spend * metrics['revenue_efficiency'] * efficiency_adj
            new_conversions += new_conv
            new_revenue += new_rev

        new_cpa = current_total['spend'] / new_conversions if new_conversions > 0 else 0
        new_roas = new_revenue / current_total['spend'] if current_total['spend'] > 0 else 0

        return {
            'spend': current_total['spend'],  # Total spend unchanged
            'conversions': new_conversions,
            'revenue': new_revenue,
            'cpa': new_cpa,
            'roas': new_roas,
        }

    def get_efficiency_ranking(
        self,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank platforms by efficiency metrics

        Returns list sorted by efficiency score
        """
        max_date = self.df['date'].max()
        start_date = max_date - pd.Timedelta(days=13)

        filtered = self._filter_data(
            start_date.strftime('%Y-%m-%d'),
            max_date.strftime('%Y-%m-%d'),
            platforms,
            geos
        )

        if len(filtered) == 0:
            return []

        rankings = []
        for platform in filtered['platform'].unique():
            platform_data = filtered[filtered['platform'] == platform]
            metrics = self.analytics.compute_metrics(platform_data)

            rankings.append({
                'platform': platform,
                'cpa': metrics['cpa'],
                'roas': metrics['roas'],
                'cvr': metrics['cvr'],
                'spend': metrics['spend'],
                'conversions': metrics['conversions'],
                'efficiency_score': (metrics['roas'] / metrics['cpa']) if metrics['cpa'] > 0 else 0,
            })

        # Sort by CPA (lower is better)
        rankings.sort(key=lambda x: x['cpa'])

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings

    def get_reallocation_recommendation(
        self,
        platforms: Optional[List[str]] = None,
        geos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate budget reallocation recommendation based on efficiency

        Returns recommended shifts from worst to best performers
        """
        rankings = self.get_efficiency_ranking(platforms, geos)

        if len(rankings) < 2:
            return {'recommendation': None, 'reason': 'insufficient_platforms'}

        best = rankings[0]
        worst = rankings[-1]

        # Only recommend if there's meaningful efficiency difference
        cpa_diff_pct = ((worst['cpa'] - best['cpa']) / best['cpa'] * 100) if best['cpa'] > 0 else 0

        if cpa_diff_pct < 15:  # Less than 15% difference
            return {
                'recommendation': None,
                'reason': 'efficiency_similar',
                'cpa_difference_pct': cpa_diff_pct,
            }

        # Recommend shifting 10-20% based on efficiency gap
        shift_pct = min(20, max(10, cpa_diff_pct / 5))

        # Simulate the shift
        scenario = self.simulate_budget_scenario(
            scenario_type='shift',
            source_platform=worst['platform'],
            target_platform=best['platform'],
            change_pct=shift_pct,
            platforms=platforms,
            geos=geos,
        )

        return {
            'recommendation': {
                'action': 'shift_budget',
                'from_platform': worst['platform'],
                'to_platform': best['platform'],
                'shift_pct': shift_pct,
                'from_cpa': worst['cpa'],
                'to_cpa': best['cpa'],
            },
            'projected_impact': scenario['impact'],
            'efficiency_rankings': rankings,
            'cpa_difference_pct': cpa_diff_pct,
        }

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

    def _empty_projection(self) -> Dict[str, Any]:
        """Return empty projection structure"""
        return {
            'metric': None,
            'method': None,
            'historical': {'dates': [], 'values': []},
            'projection': {'dates': [], 'values': [], 'lower_bound': [], 'upper_bound': []},
            'trend': {'direction': 'unknown', 'slope': 0, 'slope_pct_per_day': 0},
            'current_value': 0,
            'projected_end_value': 0,
            'projected_change_pct': 0,
        }

    def _empty_scenario(self) -> Dict[str, Any]:
        """Return empty scenario structure"""
        return {
            'scenario_type': None,
            'current': {},
            'projected': {},
            'impact': {},
        }
