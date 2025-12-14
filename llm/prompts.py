"""
LLM Prompt Templates for FE Media Intelligence Platform

These prompts ensure the LLM:
1. Only uses numbers from pre-computed analysis
2. Never hallucinates or invents data
3. Provides executive-level explanations
4. Uses Indian Rupee formatting
"""

SYSTEM_PROMPT = """You are the AI analytics assistant for FE Media Intelligence Platform, a sophisticated marketing analytics system.

CRITICAL RULES - YOU MUST FOLLOW THESE:

1. DATA GROUNDING: You will receive pre-computed analysis results as JSON. Use ONLY the numbers provided in that JSON. NEVER calculate, estimate, or invent any numbers.

2. CURRENCY: All monetary values are in Indian Rupees (₹). Use Indian number formatting:
   - ₹1,00,000 = 1 Lakh
   - ₹1,00,00,000 = 1 Crore
   - Format large numbers as "₹X.XX L" or "₹X.XX Cr"

3. CONCISENESS: Be direct and executive-friendly. Lead with the insight, not the methodology.

4. STRUCTURE: For diagnostic questions, use this structure:
   - Lead with the key finding
   - Explain the primary driver
   - Quantify the impact
   - (Optional) Suggest next steps

5. ACCURACY: If a number isn't in the provided data, say "data not available" rather than guessing.

6. TONE: Professional, confident, analytical. You're a senior analyst presenting to marketing executives.

You are NOT a generic chatbot. You are a specialized analytics reasoning layer that explains pre-computed results."""


def build_analysis_prompt(query_type: str, user_question: str, analysis_data: dict) -> str:
    """
    Build a prompt for the LLM based on query type and analysis data

    Args:
        query_type: Type of analysis (diagnostic, comparison, forecast, etc.)
        user_question: The user's original question
        analysis_data: Pre-computed analysis results as dictionary
    """
    import json

    data_json = json.dumps(analysis_data, indent=2, default=str)

    if query_type == 'diagnostic':
        return f"""User Question: {user_question}

Analysis Type: Performance Diagnostic

Pre-Computed Analysis Results:
{data_json}

Based on the analysis above, explain:
1. What happened to performance (use exact numbers from the data)
2. What is the primary driver (CPC vs CVR contribution)
3. Which platform is most responsible
4. What this means for the business

Be concise (3-5 sentences). Use ₹ for currency. Only cite numbers from the data above."""

    elif query_type == 'comparison':
        return f"""User Question: {user_question}

Analysis Type: Platform/Channel Comparison

Pre-Computed Analysis Results:
{data_json}

Based on the analysis above:
1. Compare the performance metrics across dimensions
2. Identify the best and worst performers
3. Quantify the differences
4. Provide a clear recommendation

Be direct and use exact numbers from the data. Format currency in ₹."""

    elif query_type == 'forecast':
        return f"""User Question: {user_question}

Analysis Type: Trend Projection

Pre-Computed Analysis Results:
{data_json}

Based on the analysis above:
1. Describe the current trend direction and magnitude
2. State the projected values (use the exact projections provided)
3. Note the confidence/uncertainty
4. Explain what this means for planning

Use the exact projection numbers provided. Do not extrapolate further."""

    elif query_type == 'scenario':
        return f"""User Question: {user_question}

Analysis Type: Budget Scenario Simulation

Pre-Computed Analysis Results:
{data_json}

Based on the analysis above:
1. Describe the scenario being simulated
2. State the projected impact on key metrics (conversions, CPA, ROAS)
3. Explain whether this is recommended
4. Quantify the trade-offs

Use the exact projected numbers provided. Format all currency in ₹."""

    elif query_type == 'recommendation':
        return f"""User Question: {user_question}

Analysis Type: Budget Reallocation Recommendation

Pre-Computed Analysis Results:
{data_json}

Based on the analysis above:
1. State the recommendation clearly
2. Explain why (efficiency differences)
3. Quantify the expected impact
4. Note any caveats

Be direct and actionable. Use exact numbers from the analysis."""

    elif query_type == 'lookup':
        return f"""User Question: {user_question}

Analysis Type: Data Lookup

Pre-Computed Analysis Results:
{data_json}

Answer the user's question directly using the data above.
Be concise - just provide the requested information.
Format currency in ₹ and use Indian number formatting."""

    else:  # general
        return f"""User Question: {user_question}

Pre-Computed Analysis Results:
{data_json}

Answer the user's question based on the data above.
Be concise and professional.
Use ₹ for currency and Indian number formatting.
Only use numbers from the provided data."""


def build_classification_prompt(user_question: str) -> str:
    """
    Build a prompt to classify the user's question intent
    """
    return f"""Classify this marketing analytics question into ONE category:

Question: "{user_question}"

Categories:
- diagnostic: Questions about WHY performance changed (e.g., "Why did CPA increase?", "What's driving the change?")
- comparison: Questions comparing platforms/channels/geos (e.g., "Which platform is best?", "Compare Google vs Meta")
- forecast: Questions about future trends (e.g., "What will happen next week?", "What's the trend?")
- scenario: Questions about "what if" budget changes (e.g., "What if I increase spend?", "What if I shift budget?")
- recommendation: Questions asking for advice (e.g., "Should I reallocate budget?", "Where should I invest?")
- lookup: Simple data retrieval (e.g., "What was spend in Maharashtra?", "How many conversions last week?")
- general: Other analytics questions

Respond with ONLY the category name, nothing else."""


def build_entity_extraction_prompt(user_question: str, available_platforms: list, available_geos: list) -> str:
    """
    Extract entities (platforms, geos, metrics, time periods) from the question
    """
    return f"""Extract entities from this marketing analytics question:

Question: "{user_question}"

Available Platforms: {', '.join(available_platforms)}
Available Geos: {', '.join(available_geos)}

Extract and return as JSON:
{{
    "platforms": [list of mentioned platforms or empty],
    "geos": [list of mentioned geos or empty],
    "metrics": [list of mentioned metrics like CPA, ROAS, conversions, spend],
    "time_reference": "this week" | "last week" | "last month" | null,
    "comparison_type": "wow" | "platform" | "geo" | "ad_type" | null
}}

Only include entities explicitly mentioned or clearly implied. Return valid JSON only."""
