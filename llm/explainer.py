"""
LLM Explainer - Claude API integration for natural language explanations

Ensures all explanations are grounded in pre-computed analysis results.
"""

import os
import json
from typing import Optional, Dict, Any
from anthropic import Anthropic
from .prompts import SYSTEM_PROMPT, build_analysis_prompt, build_classification_prompt, build_entity_extraction_prompt
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


class LLMExplainer:
    """
    LLM integration layer that ensures grounded explanations.
    The LLM never sees raw data - only pre-computed analysis results.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM explainer

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE

    def explain(
        self,
        query_type: str,
        user_question: str,
        analysis_data: Dict[str, Any],
    ) -> str:
        """
        Generate a natural language explanation of analysis results

        Args:
            query_type: Type of analysis (diagnostic, comparison, etc.)
            user_question: The user's original question
            analysis_data: Pre-computed analysis results (the LLM can ONLY use these numbers)

        Returns:
            Natural language explanation grounded in the analysis data
        """
        prompt = build_analysis_prompt(query_type, user_question, analysis_data)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"

    def classify_intent(self, user_question: str) -> str:
        """
        Classify the user's question intent

        Returns one of: diagnostic, comparison, forecast, scenario, recommendation, lookup, general
        """
        prompt = build_classification_prompt(user_question)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            intent = response.content[0].text.strip().lower()

            # Validate response
            valid_intents = ['diagnostic', 'comparison', 'forecast', 'scenario',
                            'recommendation', 'lookup', 'general']
            if intent in valid_intents:
                return intent

            # Try to extract from response if it contains extra text
            for valid in valid_intents:
                if valid in intent:
                    return valid

            return 'general'

        except Exception as e:
            # Default to diagnostic for most common use case
            return 'diagnostic'

    def extract_entities(
        self,
        user_question: str,
        available_platforms: list,
        available_geos: list,
    ) -> Dict[str, Any]:
        """
        Extract entities (platforms, geos, metrics, time) from the question

        Returns structured extraction for routing logic
        """
        prompt = build_entity_extraction_prompt(
            user_question, available_platforms, available_geos
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            text = response.content[0].text.strip()

            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()

            return json.loads(text)

        except (json.JSONDecodeError, Exception):
            # Return empty extraction if parsing fails
            return {
                'platforms': [],
                'geos': [],
                'metrics': [],
                'time_reference': None,
                'comparison_type': None,
            }

    def generate_summary(
        self,
        metrics: Dict[str, Any],
        period_description: str,
    ) -> str:
        """
        Generate a brief summary of current performance

        Args:
            metrics: Current period metrics
            period_description: e.g., "Last 7 days"

        Returns:
            2-3 sentence summary
        """
        prompt = f"""Summarize this marketing performance data in 2-3 sentences:

Period: {period_description}

Metrics:
- Total Spend: ₹{metrics.get('spend', 0):,.0f}
- Conversions: {metrics.get('conversions', 0):,.0f}
- CPA: ₹{metrics.get('cpa', 0):,.0f}
- ROAS: {metrics.get('roas', 0):.2f}x
- CTR: {metrics.get('ctr', 0):.2f}%
- CVR: {metrics.get('cvr', 0):.2f}%

Focus on headline performance. Use Indian Rupee formatting (₹, L for lakhs, Cr for crores).
Be concise and executive-friendly."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            return f"Performance summary unavailable: {str(e)}"


class MockLLMExplainer:
    """
    Mock LLM explainer for testing without API calls.
    Generates structured but not AI-generated responses.
    """

    def explain(
        self,
        query_type: str,
        user_question: str,
        analysis_data: Dict[str, Any],
    ) -> str:
        """Generate mock explanation"""
        if query_type == 'diagnostic':
            return self._mock_diagnostic(analysis_data)
        elif query_type == 'comparison':
            return self._mock_comparison(analysis_data)
        elif query_type == 'forecast':
            return self._mock_forecast(analysis_data)
        elif query_type == 'scenario':
            return self._mock_scenario(analysis_data)
        else:
            return "Analysis complete. Please review the data in the dashboard."

    def classify_intent(self, user_question: str) -> str:
        """Simple keyword-based classification"""
        q = user_question.lower()

        if any(w in q for w in ['why', 'driver', 'cause', 'reason', 'what happened']):
            return 'diagnostic'
        elif any(w in q for w in ['compare', 'vs', 'versus', 'better', 'best', 'worst']):
            return 'comparison'
        elif any(w in q for w in ['forecast', 'predict', 'trend', 'next week', 'will']):
            return 'forecast'
        elif any(w in q for w in ['what if', 'scenario', 'shift', 'increase', 'decrease']):
            return 'scenario'
        elif any(w in q for w in ['recommend', 'should', 'advice', 'suggest']):
            return 'recommendation'
        elif any(w in q for w in ['what was', 'how many', 'show me', 'total']):
            return 'lookup'
        else:
            return 'general'

    def extract_entities(
        self,
        user_question: str,
        available_platforms: list,
        available_geos: list,
    ) -> Dict[str, Any]:
        """Simple entity extraction"""
        q = user_question.lower()

        platforms = [p for p in available_platforms if p.lower() in q]
        geos = [g for g in available_geos if g.lower() in q]

        return {
            'platforms': platforms,
            'geos': geos,
            'metrics': [],
            'time_reference': 'this week' if 'this week' in q else 'last week' if 'last week' in q else None,
            'comparison_type': None,
        }

    def _mock_diagnostic(self, data: Dict) -> str:
        if 'cpa_decomposition' in data:
            decomp = data['cpa_decomposition']
            change = decomp.get('total_change', 0)
            driver = decomp.get('primary_driver', 'unknown')

            if change > 0:
                return f"CPA increased by ₹{abs(change):.0f}. The primary driver is {driver}. Review the diagnostic breakdown for details."
            else:
                return f"CPA decreased by ₹{abs(change):.0f}. Performance is improving, primarily driven by {driver}."
        return "Diagnostic analysis complete. Review the data for details."

    def _mock_comparison(self, data: Dict) -> str:
        return "Platform comparison complete. See the breakdown table for detailed metrics."

    def _mock_forecast(self, data: Dict) -> str:
        if 'trend' in data:
            direction = data['trend'].get('direction', 'stable')
            return f"The trend is {direction}. See the projection chart for details."
        return "Forecast generated. Review the projection chart."

    def _mock_scenario(self, data: Dict) -> str:
        if 'impact' in data:
            impact = data['impact']
            conv_change = impact.get('conversions_change_pct', 0)
            return f"This scenario would result in a {conv_change:.1f}% change in conversions."
        return "Scenario simulation complete. Review the projected impact."

    def generate_summary(self, metrics: Dict, period: str) -> str:
        spend = metrics.get('spend', 0)
        cpa = metrics.get('cpa', 0)
        roas = metrics.get('roas', 0)
        return f"In {period}: Total spend ₹{spend:,.0f}, CPA ₹{cpa:.0f}, ROAS {roas:.2f}x."
