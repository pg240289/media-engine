"""
LLM Prompt Templates for FE Media Intelligence Platform

These prompts ensure the LLM:
1. Only uses numbers from pre-computed analysis
2. Never hallucinates or invents data
3. Provides executive-level explanations
4. Uses Indian Rupee formatting
"""

SYSTEM_PROMPT = """You are an AI analytics assistant for a marketing intelligence platform. You help marketing executives understand their campaign performance.

CRITICAL RULES:
1. Use ONLY the numbers provided in the data. Never invent or estimate numbers.
2. Format currency in Indian Rupees (₹): ₹1,00,000 = 1 Lakh (L), ₹1,00,00,000 = 1 Crore (Cr)
3. Be concise but insightful - lead with the key finding, then explain why it matters.
4. If asked about campaigns, analyze CAMPAIGNS. If asked about platforms, analyze PLATFORMS. Match your answer to what was asked.
5. Provide actionable recommendations when relevant.

Industry Benchmarks:
- CPA below ₹500 = Strong performance
- ROAS above 3x = Healthy returns
- CTR above 1% = Good engagement"""


def build_analysis_prompt(query_type: str, user_question: str, analysis_data: dict) -> str:
    """
    Build a focused prompt that directs the LLM to the right data.
    """
    import json

    question_lower = user_question.lower()

    # Determine what the user is asking about and extract the relevant data
    focus_data = {}
    focus_instruction = ""

    # Always include summary and period info
    focus_data['summary'] = analysis_data.get('summary', {})
    focus_data['period'] = analysis_data.get('period', {})

    # Determine the primary focus based on the question
    if any(word in question_lower for word in ['campaign', 'campaigns']):
        focus_data['campaign_breakdown'] = analysis_data.get('campaign_breakdown', [])
        focus_data['best_campaign_by_cpa'] = analysis_data.get('best_campaign_by_cpa', {})
        focus_data['worst_campaign_by_cpa'] = analysis_data.get('worst_campaign_by_cpa', {})
        focus_instruction = """
FOCUS: This question is about CAMPAIGNS. Use the campaign_breakdown data.
- Each entry in campaign_breakdown represents a different campaign
- The 'campaign_name' field identifies each campaign
- Find the best/worst performing campaigns based on the metrics asked about"""

    elif any(word in question_lower for word in ['client', 'clients', 'account', 'accounts', 'advertiser', 'brand']):
        focus_data['client_breakdown'] = analysis_data.get('client_breakdown', [])
        focus_instruction = """
FOCUS: This question is about CLIENTS/ADVERTISERS. Use the client_breakdown data.
- Each entry represents a different client/advertiser
- The 'client' field identifies each client"""

    elif any(word in question_lower for word in ['geo', 'geography', 'region', 'state', 'location', 'maharashtra', 'karnataka', 'tamil', 'delhi', 'gujarat']):
        focus_data['geo_breakdown'] = analysis_data.get('geo_breakdown', [])
        focus_instruction = """
FOCUS: This question is about GEOGRAPHIC REGIONS. Use the geo_breakdown data.
- Each entry represents a different geographic region
- The 'geo' field identifies each region"""

    elif any(word in question_lower for word in ['creative', 'creatives', 'ad creative', 'creative type']):
        focus_data['creative_breakdown'] = analysis_data.get('creative_breakdown', [])
        focus_instruction = """
FOCUS: This question is about CREATIVES. Use the creative_breakdown data.
- Each entry represents a different creative type
- The 'creative_type' field identifies each creative"""

    elif any(word in question_lower for word in ['ad type', 'ad format', 'format', 'ad types']):
        focus_data['ad_type_breakdown'] = analysis_data.get('ad_type_breakdown', [])
        focus_instruction = """
FOCUS: This question is about AD TYPES/FORMATS. Use the ad_type_breakdown data.
- Each entry represents a different ad type
- The 'ad_type' field identifies each ad type"""

    elif any(word in question_lower for word in ['objective', 'objectives', 'goal', 'goals']):
        focus_data['objective_breakdown'] = analysis_data.get('objective_breakdown', [])
        focus_instruction = """
FOCUS: This question is about CAMPAIGN OBJECTIVES. Use the objective_breakdown data.
- Each entry represents a different objective (Conversions, Traffic, etc.)
- The 'objective' field identifies each objective"""

    elif any(word in question_lower for word in ['why', 'change', 'increased', 'decreased', 'dropped', 'rose', 'driving', 'cause']):
        focus_data['period_comparison'] = analysis_data.get('period_comparison', {})
        focus_data['platform_breakdown'] = analysis_data.get('platform_breakdown', [])
        focus_instruction = """
FOCUS: This is a DIAGNOSTIC question about changes. Use the period_comparison data.
- Compare current vs previous period metrics
- Identify what changed and why
- Use platform_breakdown to attribute changes to specific platforms"""

    elif any(word in question_lower for word in ['platform', 'platforms', 'google', 'meta', 'amazon', 'dv360', 'facebook']):
        focus_data['platform_breakdown'] = analysis_data.get('platform_breakdown', [])
        focus_instruction = """
FOCUS: This question is about PLATFORMS. Use the platform_breakdown data.
- Each entry represents a different advertising platform
- The 'platform' field identifies each platform (Google Ads, Meta, Amazon Ads, DV360)"""

    elif any(word in question_lower for word in ['trend', 'forecast', 'predict', 'next week', 'projection']):
        focus_data['daily_trend'] = analysis_data.get('daily_trend', [])
        focus_data['period_comparison'] = analysis_data.get('period_comparison', {})
        focus_instruction = """
FOCUS: This is a TREND/FORECAST question. Use the daily_trend data.
- Analyze the trajectory of metrics over time
- Note whether metrics are improving or declining"""

    elif any(word in question_lower for word in ['best', 'top', 'highest', 'winner', 'performing']):
        # Determine what "best" refers to
        if 'campaign' in question_lower:
            focus_data['campaign_breakdown'] = analysis_data.get('campaign_breakdown', [])
            focus_data['best_campaign_by_cpa'] = analysis_data.get('best_campaign_by_cpa', {})
            focus_instruction = "FOCUS: Find the BEST CAMPAIGN. Use campaign_breakdown data."
        elif 'client' in question_lower:
            focus_data['client_breakdown'] = analysis_data.get('client_breakdown', [])
            focus_instruction = "FOCUS: Find the BEST CLIENT. Use client_breakdown data."
        elif 'geo' in question_lower or 'region' in question_lower:
            focus_data['geo_breakdown'] = analysis_data.get('geo_breakdown', [])
            focus_instruction = "FOCUS: Find the BEST GEO. Use geo_breakdown data."
        else:
            focus_data['platform_breakdown'] = analysis_data.get('platform_breakdown', [])
            focus_instruction = "FOCUS: Find the BEST PLATFORM. Use platform_breakdown data."

    elif any(word in question_lower for word in ['worst', 'bottom', 'lowest', 'underperforming', 'poor']):
        # Similar logic for "worst"
        if 'campaign' in question_lower:
            focus_data['campaign_breakdown'] = analysis_data.get('campaign_breakdown', [])
            focus_data['worst_campaign_by_cpa'] = analysis_data.get('worst_campaign_by_cpa', {})
            focus_instruction = "FOCUS: Find the WORST CAMPAIGN. Use campaign_breakdown data."
        else:
            focus_data['platform_breakdown'] = analysis_data.get('platform_breakdown', [])
            focus_instruction = "FOCUS: Find the WORST PLATFORM. Use platform_breakdown data."

    else:
        # Default: include platform breakdown and summary
        focus_data['platform_breakdown'] = analysis_data.get('platform_breakdown', [])
        focus_data['period_comparison'] = analysis_data.get('period_comparison', {})
        focus_instruction = """
FOCUS: General performance question. Use platform_breakdown for overview.
- Provide a high-level summary of performance
- Highlight key insights and recommendations"""

    data_json = json.dumps(focus_data, indent=2, default=str)

    return f"""User Question: "{user_question}"

{focus_instruction}

DATA:
{data_json}

RESPONSE GUIDELINES:
1. Answer the question directly - lead with the key finding
2. Use ONLY numbers from the data above
3. Format currency as ₹X.XX L (lakhs) or ₹X.XX Cr (crores)
4. Explain WHY this matters (not just what the numbers are)
5. Provide 1-2 actionable recommendations if relevant
6. Keep response concise but valuable"""


def build_conversational_prompt(
    query_type: str,
    user_question: str,
    analysis_data: dict,
    conversation_history: list = None
) -> str:
    """
    Build a prompt that includes conversation history for follow-up questions.
    """
    base_prompt = build_analysis_prompt(query_type, user_question, analysis_data)

    if not conversation_history:
        return base_prompt

    # Build conversation context (keep it brief)
    history_text = "\n--- PREVIOUS CONVERSATION ---\n"
    for turn in conversation_history[-3:]:  # Last 3 turns only
        history_text += f"Q: {turn.get('question', '')}\n"
        # Truncate long answers
        answer = turn.get('answer_summary', turn.get('answer', ''))
        if len(answer) > 300:
            answer = answer[:300] + "..."
        history_text += f"A: {answer}\n\n"
    history_text += "--- END HISTORY ---\n"

    return f"""{history_text}

{base_prompt}

Note: This is a follow-up question. Reference previous context if relevant, but focus on answering the current question."""


def build_classification_prompt(user_question: str) -> str:
    """
    Build a prompt to classify the user's question intent
    """
    return f"""Classify this marketing analytics question into ONE category:

Question: "{user_question}"

Categories:
- diagnostic: Questions about WHY performance changed
- comparison: Questions comparing different things
- forecast: Questions about future trends
- recommendation: Questions asking for advice
- lookup: Simple data retrieval questions
- general: Other questions

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
    "comparison_type": "platform" | "campaign" | "client" | "geo" | "ad_type" | "creative" | "objective" | null
}}

Rules:
- "campaign" or "campaigns" → comparison_type: "campaign"
- "client" or "account" or "advertiser" → comparison_type: "client"
- "creative" → comparison_type: "creative"
- "platform" or specific platform names → comparison_type: "platform"
- "geo" or "region" or specific state names → comparison_type: "geo"

Return valid JSON only."""
