"""
Query Router - Natural language query processing and orchestration

Routes user questions to appropriate analysis engines.
All analysis is computed before passing to LLM for explanation.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .analytics import AnalyticsEngine
from .diagnostics import DiagnosticsEngine
from .forecasting import ForecastingEngine
from llm.explainer import LLMExplainer, MockLLMExplainer


class QueryRouter:
    """
    Routes natural language queries to appropriate analysis engines.

    Flow:
    1. Classify query intent
    2. Extract entities (platforms, geos, time)
    3. Run appropriate analysis
    4. Pass results to LLM for explanation
    """

    def __init__(
        self,
        df: pd.DataFrame,
        llm_api_key: Optional[str] = None,
        use_mock_llm: bool = False,
    ):
        """
        Initialize the query router

        Args:
            df: Performance data DataFrame
            llm_api_key: Anthropic API key
            use_mock_llm: If True, use mock LLM for testing
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Initialize engines
        self.analytics = AnalyticsEngine(df)
        self.diagnostics = DiagnosticsEngine(df)
        self.forecasting = ForecastingEngine(df)

        # Initialize LLM
        if use_mock_llm:
            self.llm = MockLLMExplainer()
        else:
            try:
                self.llm = LLMExplainer(api_key=llm_api_key)
            except ValueError:
                # Fall back to mock if no API key
                self.llm = MockLLMExplainer()

        # Get available values for entity extraction
        self.available_platforms = self.df['platform'].unique().tolist()
        self.available_geos = self.df['geo'].unique().tolist()

    def process_query(
        self,
        user_question: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a natural language query

        Args:
            user_question: The user's question
            filters: Optional filters already applied (platforms, geos, date range)
            conversation_history: Optional list of previous Q&A pairs for context

        Returns:
            Dictionary with:
            - intent: Classified intent
            - analysis: Computed analysis results
            - explanation: LLM-generated explanation
            - visualization_hint: Suggestion for UI visualization
        """
        # Step 1: Classify intent
        intent = self.llm.classify_intent(user_question)

        # Step 2: Extract entities
        entities = self.llm.extract_entities(
            user_question,
            self.available_platforms,
            self.available_geos
        )

        # Merge with provided filters
        # UI filters ALWAYS take precedence - they define the dataset the user is viewing
        # Question-extracted entities are only used when no UI filter is set
        if filters:
            # UI platform filter takes precedence
            if filters.get('platforms'):
                entities['platforms'] = filters['platforms']
            # UI geo filter takes precedence
            if filters.get('geos'):
                entities['geos'] = filters['geos']

        # Step 3: Compute date ranges - USE FILTERS if provided
        date_ranges = self._compute_date_ranges(
            time_reference=entities.get('time_reference'),
            filters=filters
        )

        # Step 4: Run appropriate analysis
        analysis_data = self._run_analysis(
            intent,
            entities,
            date_ranges,
            user_question
        )

        # Step 5: Generate explanation with conversation context
        explanation = self.llm.explain(
            query_type=intent,
            user_question=user_question,
            analysis_data=analysis_data,
            conversation_history=conversation_history
        )

        # Step 6: Determine visualization hint
        visualization = self._get_visualization_hint(intent, analysis_data)

        return {
            'intent': intent,
            'entities': entities,
            'analysis': analysis_data,
            'explanation': explanation,
            'visualization_hint': visualization,
            'date_ranges': date_ranges,
        }

    def _compute_date_ranges(
        self,
        time_reference: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Compute date ranges based on filters or time reference.

        Priority:
        1. Use filters['start_date'] and filters['end_date'] if provided
        2. Fall back to time_reference interpretation
        3. Default to last 7 days
        """
        # If filters provide explicit date range, use that
        if filters and filters.get('start_date') and filters.get('end_date'):
            # Convert to datetime if needed
            start = filters['start_date']
            end = filters['end_date']

            if hasattr(start, 'strftime'):
                current_start = pd.Timestamp(start)
            else:
                current_start = pd.to_datetime(start)

            if hasattr(end, 'strftime'):
                current_end = pd.Timestamp(end)
            else:
                current_end = pd.to_datetime(end)

            # Calculate the period length to create a matching previous period
            period_days = (current_end - current_start).days
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=period_days)

            return {
                'current_start': current_start.strftime('%Y-%m-%d'),
                'current_end': current_end.strftime('%Y-%m-%d'),
                'previous_start': previous_start.strftime('%Y-%m-%d'),
                'previous_end': previous_end.strftime('%Y-%m-%d'),
            }

        # Fall back to original logic if no filter dates
        max_date = self.df['date'].max()

        if time_reference == 'last month':
            current_end = max_date
            current_start = max_date - timedelta(days=29)
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=29)
        else:  # Default to last 7 days vs previous 7 days
            current_end = max_date
            current_start = max_date - timedelta(days=6)
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=6)

        return {
            'current_start': current_start.strftime('%Y-%m-%d'),
            'current_end': current_end.strftime('%Y-%m-%d'),
            'previous_start': previous_start.strftime('%Y-%m-%d'),
            'previous_end': previous_end.strftime('%Y-%m-%d'),
        }

    def _run_analysis(
        self,
        intent: str,
        entities: Dict[str, Any],
        date_ranges: Dict[str, str],
        user_question: str,
    ) -> Dict[str, Any]:
        """
        Run COMPREHENSIVE analysis - gather all relevant data and let LLM decide what to use.

        This approach eliminates brittle routing logic by always providing rich context.
        The LLM will select the relevant data based on the user's actual question.
        """
        platforms = entities.get('platforms') or None
        geos = entities.get('geos') or None

        try:
            # Always run comprehensive analysis regardless of intent
            return self._run_comprehensive_analysis(
                date_ranges, platforms, geos, user_question
            )
        except Exception as e:
            # Return a minimal analysis result with error info
            return {
                'error': str(e),
                'summary': self.analytics.get_summary_metrics(
                    start_date=date_ranges['current_start'],
                    end_date=date_ranges['current_end'],
                    platforms=platforms,
                    geos=geos,
                ),
                'period': date_ranges,
            }

    def _run_comprehensive_analysis(
        self,
        date_ranges: Dict,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
        user_question: str,
    ) -> Dict[str, Any]:
        """
        Gather ALL available breakdowns and metrics.
        Let the LLM decide what's relevant based on the question.
        """
        result = {
            'period': date_ranges,
            'user_question': user_question,
        }

        # 1. Overall summary metrics
        result['summary'] = self.analytics.get_summary_metrics(
            start_date=date_ranges['current_start'],
            end_date=date_ranges['current_end'],
            platforms=platforms,
            geos=geos,
        )

        # 2. Period-over-period comparison (diagnostic data)
        try:
            diagnostic = self.diagnostics.analyze_period_change(
                current_start=date_ranges['current_start'],
                current_end=date_ranges['current_end'],
                previous_start=date_ranges['previous_start'],
                previous_end=date_ranges['previous_end'],
                platforms=platforms,
                geos=geos,
            )
            result['period_comparison'] = diagnostic
        except Exception:
            pass

        # 3. Breakdowns by ALL available dimensions
        dimensions_to_try = [
            ('platform', 'platform_breakdown'),
            ('campaign_name', 'campaign_breakdown'),
            ('client', 'client_breakdown'),
            ('geo', 'geo_breakdown'),
            ('ad_type', 'ad_type_breakdown'),
            ('objective', 'objective_breakdown'),
            ('creative_type', 'creative_breakdown'),
        ]

        for column, result_key in dimensions_to_try:
            if column in self.df.columns:
                try:
                    breakdown = self.analytics.get_breakdown_by_dimension(
                        dimension=column,
                        start_date=date_ranges['current_start'],
                        end_date=date_ranges['current_end'],
                        platforms=platforms,
                        geos=geos,
                    )
                    if len(breakdown) > 0:
                        # Limit to top 15 to avoid token bloat
                        result[result_key] = breakdown.head(15).to_dict('records')
                except Exception:
                    pass

        # 4. Time series for trends (if forecasting might be relevant)
        try:
            time_series = self.analytics.get_time_series(
                metric='conversions',
                granularity='daily',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            if len(time_series) > 0:
                result['daily_trend'] = time_series.tail(14).to_dict('records')
        except Exception:
            pass

        # 5. Top/bottom performers summary
        try:
            # Find best and worst by CPA
            if 'campaign_breakdown' in result and result['campaign_breakdown']:
                campaigns = result['campaign_breakdown']
                campaigns_with_spend = [c for c in campaigns if c.get('spend', 0) > 0 and c.get('conversions', 0) > 0]
                if campaigns_with_spend:
                    best_cpa = min(campaigns_with_spend, key=lambda x: x.get('cpa', float('inf')))
                    worst_cpa = max(campaigns_with_spend, key=lambda x: x.get('cpa', 0))
                    result['best_campaign_by_cpa'] = best_cpa
                    result['worst_campaign_by_cpa'] = worst_cpa
        except Exception:
            pass

        return result

    def _run_diagnostic_analysis(
        self,
        date_ranges: Dict,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run diagnostic analysis"""
        analysis = self.diagnostics.analyze_period_change(
            current_start=date_ranges['current_start'],
            current_end=date_ranges['current_end'],
            previous_start=date_ranges['previous_start'],
            previous_end=date_ranges['previous_end'],
            platforms=platforms,
            geos=geos,
        )

        return analysis

    def _run_comparison_analysis(
        self,
        date_ranges: Dict,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run comparison analysis"""
        # Determine comparison dimension
        comparison_type = entities.get('comparison_type')

        # Map comparison_type to actual column name, with validation
        dimension_mapping = {
            'campaign': 'campaign_name',
            'client': 'client',
            'geo': 'geo',
            'ad_type': 'ad_type',
            'creative': 'creative_type',
            'objective': 'objective',
            'platform': 'platform',
        }

        # Determine dimension based on comparison_type
        dimension = 'platform'  # default fallback

        if comparison_type and comparison_type in dimension_mapping:
            mapped_column = dimension_mapping[comparison_type]
            if mapped_column in self.df.columns:
                dimension = mapped_column
            else:
                # Column doesn't exist, fall back to platform
                dimension = 'platform'
        elif comparison_type == 'geo' or geos:
            dimension = 'geo'

        breakdown = self.analytics.get_breakdown_by_dimension(
            dimension=dimension,
            start_date=date_ranges['current_start'],
            end_date=date_ranges['current_end'],
            platforms=platforms,
            geos=geos,
        )

        comparison = self.analytics.get_comparison(
            dimension=dimension,
            current_start=date_ranges['current_start'],
            current_end=date_ranges['current_end'],
            previous_start=date_ranges['previous_start'],
            previous_end=date_ranges['previous_end'],
            platforms=platforms,
            geos=geos,
        )

        return {
            'dimension': dimension,
            'breakdown': breakdown.to_dict('records') if len(breakdown) > 0 else [],
            'comparison': comparison.to_dict('records') if len(comparison) > 0 else [],
            'period': date_ranges,
        }

    def _run_forecast_analysis(
        self,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run forecast analysis"""
        # Determine metric to forecast
        metrics_mentioned = entities.get('metrics', [])

        if 'cpa' in [m.lower() for m in metrics_mentioned]:
            metric = 'cpa'
        elif 'conversions' in [m.lower() for m in metrics_mentioned]:
            metric = 'conversions'
        elif 'roas' in [m.lower() for m in metrics_mentioned]:
            metric = 'roas'
        else:
            metric = 'conversions'  # Default

        projection = self.forecasting.project_trend(
            metric=metric,
            periods_ahead=7,
            lookback_days=21,
            platforms=platforms,
            geos=geos,
        )

        return projection

    def _run_scenario_analysis(
        self,
        user_question: str,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run scenario simulation"""
        q = user_question.lower()

        # Detect scenario type from question
        if 'shift' in q or 'move' in q or 'reallocate' in q:
            # Try to detect source and target platforms
            source = None
            target = None

            for p in self.available_platforms:
                p_lower = p.lower()
                if f'from {p_lower}' in q or f'away from {p_lower}' in q:
                    source = p
                elif f'to {p_lower}' in q or f'into {p_lower}' in q:
                    target = p

            if source and target:
                return self.forecasting.simulate_budget_scenario(
                    scenario_type='shift',
                    source_platform=source,
                    target_platform=target,
                    change_pct=10.0,
                    platforms=platforms,
                    geos=geos,
                )

        # Default: increase scenario
        if 'decrease' in q or 'reduce' in q or 'cut' in q:
            scenario_type = 'decrease_all'
        else:
            scenario_type = 'increase_all'

        # Try to extract percentage
        change_pct = 10.0  # Default
        import re
        pct_match = re.search(r'(\d+)\s*%', q)
        if pct_match:
            change_pct = float(pct_match.group(1))

        return self.forecasting.simulate_budget_scenario(
            scenario_type=scenario_type,
            change_pct=change_pct,
            platforms=platforms,
            geos=geos,
        )

    def _run_recommendation_analysis(
        self,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run recommendation analysis"""
        return self.forecasting.get_reallocation_recommendation(
            platforms=platforms,
            geos=geos,
        )

    def _run_lookup_analysis(
        self,
        date_ranges: Dict,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run simple lookup analysis with comprehensive breakdowns"""
        summary = self.analytics.get_summary_metrics(
            start_date=date_ranges['current_start'],
            end_date=date_ranges['current_end'],
            platforms=platforms,
            geos=geos,
        )

        result = {
            'summary': summary,
            'period': date_ranges,
        }

        # Add platform breakdown
        if 'platform' in self.df.columns:
            platform_breakdown = self.analytics.get_breakdown_by_dimension(
                dimension='platform',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            result['platform_breakdown'] = platform_breakdown.to_dict('records') if len(platform_breakdown) > 0 else []

        # Add client breakdown - critical for client-related questions
        if 'client' in self.df.columns:
            client_breakdown = self.analytics.get_breakdown_by_dimension(
                dimension='client',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            result['client_breakdown'] = client_breakdown.to_dict('records') if len(client_breakdown) > 0 else []

        # Add geo breakdown
        if 'geo' in self.df.columns:
            geo_breakdown = self.analytics.get_breakdown_by_dimension(
                dimension='geo',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            result['geo_breakdown'] = geo_breakdown.to_dict('records') if len(geo_breakdown) > 0 else []

        return result

    def _run_general_analysis(
        self,
        date_ranges: Dict,
        platforms: Optional[List[str]],
        geos: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run general analysis for unclassified questions with comprehensive breakdowns"""
        summary = self.analytics.get_summary_metrics(
            start_date=date_ranges['current_start'],
            end_date=date_ranges['current_end'],
            platforms=platforms,
            geos=geos,
        )

        diagnostic = self.diagnostics.analyze_period_change(
            current_start=date_ranges['current_start'],
            current_end=date_ranges['current_end'],
            previous_start=date_ranges['previous_start'],
            previous_end=date_ranges['previous_end'],
            platforms=platforms,
            geos=geos,
        )

        result = {
            'summary': summary,
            'diagnostic': diagnostic,
            'period': date_ranges,
        }

        # Add client breakdown for client-related questions (if column exists)
        if 'client' in self.df.columns:
            client_breakdown = self.analytics.get_breakdown_by_dimension(
                dimension='client',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            result['client_breakdown'] = client_breakdown.to_dict('records') if len(client_breakdown) > 0 else []

        # Add platform breakdown (if column exists)
        if 'platform' in self.df.columns:
            platform_breakdown = self.analytics.get_breakdown_by_dimension(
                dimension='platform',
                start_date=date_ranges['current_start'],
                end_date=date_ranges['current_end'],
                platforms=platforms,
                geos=geos,
            )
            result['platform_breakdown'] = platform_breakdown.to_dict('records') if len(platform_breakdown) > 0 else []

        return result

    def _get_visualization_hint(
        self,
        intent: str,
        analysis_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Suggest visualization based on intent and data"""

        if intent == 'diagnostic':
            return {
                'type': 'diagnostic_panel',
                'show_decomposition': True,
                'show_platform_attribution': True,
            }

        elif intent == 'comparison':
            return {
                'type': 'comparison_table',
                'dimension': analysis_data.get('dimension', 'platform'),
                'show_chart': True,
            }

        elif intent == 'forecast':
            return {
                'type': 'trend_chart',
                'show_projection': True,
                'show_confidence': True,
            }

        elif intent == 'scenario':
            return {
                'type': 'scenario_comparison',
                'show_before_after': True,
            }

        elif intent == 'recommendation':
            return {
                'type': 'recommendation_card',
                'show_efficiency_ranking': True,
            }

        else:
            return {
                'type': 'summary_metrics',
                'show_table': True,
            }

    def get_suggested_questions(self) -> List[str]:
        """Get contextually relevant suggested questions"""
        # Compute what's interesting in the data
        max_date = self.df['date'].max()
        current_start = max_date - timedelta(days=6)
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=6)

        analysis = self.diagnostics.analyze_period_change(
            current_start=current_start.strftime('%Y-%m-%d'),
            current_end=max_date.strftime('%Y-%m-%d'),
            previous_start=previous_start.strftime('%Y-%m-%d'),
            previous_end=previous_end.strftime('%Y-%m-%d'),
        )

        questions = []

        # Add diagnostic question if there's a change
        if 'cpa_decomposition' in analysis:
            change = analysis['cpa_decomposition'].get('total_change_pct', 0)
            if abs(change) > 5:
                if change > 0:
                    questions.append("Why did CPA increase this week?")
                else:
                    questions.append("What's driving the CPA improvement?")

        # Add platform question
        if analysis.get('platform_attribution'):
            top_platform = analysis['platform_attribution'][0]['platform']
            questions.append(f"How is {top_platform} performing?")

        # Add standard questions
        questions.extend([
            "Which platform has the best ROAS?",
            "Should I reallocate budget between platforms?",
            "What's the conversion trend over the last 3 weeks?",
        ])

        return questions[:5]  # Return top 5
