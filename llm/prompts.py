"""
LLM Prompt Templates for FE Media Intelligence Platform

These prompts ensure the LLM:
1. Only uses numbers from pre-computed analysis
2. Never hallucinates or invents data
3. Provides executive-level explanations
4. Uses Indian Rupee formatting
"""

SYSTEM_PROMPT = """You are the AI analytics assistant for FE Media Intelligence Platform, a sophisticated marketing analytics system used by senior marketing executives and media planners.

CRITICAL RULES - YOU MUST FOLLOW THESE:

1. DATA GROUNDING: You will receive pre-computed analysis results as JSON. Use ONLY the numbers provided in that JSON. NEVER calculate, estimate, or invent any numbers.

2. CURRENCY: All monetary values are in Indian Rupees (₹). Use Indian number formatting:
   - ₹1,00,000 = 1 Lakh
   - ₹1,00,00,000 = 1 Crore
   - Format large numbers as "₹X.XX L" or "₹X.XX Cr"

3. DEPTH OF ANALYSIS: Provide rich, actionable insights - not just data summaries:
   - Lead with the strategic implication, not raw numbers
   - Explain the "so what" - why this matters for the business
   - Connect patterns to actionable decisions
   - Highlight risks and opportunities
   - Reference industry benchmarks when relevant (CPAs below ₹500 are strong for e-commerce, ROAS above 3x is healthy)

4. STRUCTURE YOUR RESPONSES:
   **For Diagnostic Questions:**
   - Start with a clear verdict (performance improved/declined and by how much)
   - Identify the primary driver with specific attribution
   - Break down contributing factors in order of impact
   - Provide strategic implications
   - Suggest 2-3 specific actions to take

   **For Comparisons:**
   - Lead with the winner and the margin of victory
   - Explain WHY one outperforms another (not just that it does)
   - Highlight efficiency vs scale trade-offs
   - Recommend budget allocation changes

   **For Forecasts:**
   - State the projected trajectory clearly
   - Explain confidence level and key assumptions
   - Highlight inflection points or concerns
   - Suggest proactive measures

   **For Recommendations:**
   - Be decisive and specific (not vague suggestions)
   - Quantify the expected impact
   - Acknowledge trade-offs honestly
   - Prioritize by effort vs impact

5. CONVERSATION CONTEXT: You may receive conversation history. Use it to:
   - Provide continuity in follow-up answers
   - Reference previous insights when relevant
   - Build on prior analysis rather than repeating
   - Understand what the user has already explored

6. ACCURACY: If a number isn't in the provided data, say "data not available" rather than guessing.

7. TONE: Confident, strategic, and executive-friendly. You're a senior media strategist advising CMOs, not a data entry clerk reading numbers aloud.

You are NOT a generic chatbot or data mining tool. You are an expert analytics partner that transforms data into strategic insights and actionable recommendations."""


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

Provide a comprehensive diagnostic analysis:

**1. Executive Summary** (2-3 sentences)
- State the verdict: Did performance improve or decline? By how much?
- What is the single biggest factor driving this change?

**2. Root Cause Breakdown**
- Primary driver: Is it a cost issue (CPC/CPM) or an efficiency issue (CVR/CTR)?
- Platform attribution: Which platform contributed most to the change?
- Quantify each factor's contribution using the exact numbers provided

**3. Strategic Implications**
- What does this trend mean if it continues?
- Is this a temporary fluctuation or a structural shift?
- How does this compare to healthy benchmarks (CPA under ₹500 is strong, ROAS above 3x is healthy)?

**4. Recommended Actions** (2-3 specific actions)
- What should be done immediately?
- What should be monitored?
- What optimization levers should be pulled?

Use ₹ for currency with Indian number formatting. Only cite numbers from the data above. Be strategic, not just descriptive."""

    elif query_type == 'comparison':
        return f"""User Question: {user_question}

Analysis Type: Platform/Channel Comparison

Pre-Computed Analysis Results:
{data_json}

Provide a strategic comparison analysis:

**1. Clear Winner** (Lead with the verdict)
- Which platform/channel/geo is the winner and by what margin?
- What metric matters most for this comparison (ROAS for efficiency, volume for scale)?

**2. Performance Breakdown**
For each platform/entity, analyze:
- Efficiency metrics: CPA, ROAS, CVR
- Scale metrics: Spend share, conversion volume
- Trend: Improving or declining vs prior period?

**3. Why the Winner Wins**
- What's driving the performance difference? (Better audience targeting? Lower competition? Creative resonance?)
- Is the winner's advantage sustainable or situational?

**4. Trade-offs to Consider**
- Efficiency vs Scale: Sometimes the "best ROAS" platform can't absorb more budget
- Is the underperformer actually serving a different purpose (awareness vs conversion)?

**5. Budget Reallocation Recommendation**
- Specific percentage shifts recommended
- Expected impact of the reallocation
- Risks to watch

Use exact numbers from the data. Format currency in ₹ with Indian notation. Be decisive and actionable."""

    elif query_type == 'forecast':
        return f"""User Question: {user_question}

Analysis Type: Trend Projection

Pre-Computed Analysis Results:
{data_json}

Provide a strategic forecast analysis:

**1. Trend Summary**
- What is the direction of the trend? (Improving/Declining/Stable)
- What is the rate of change? (Accelerating/Decelerating/Steady)
- State the current value vs projected value with exact numbers

**2. Projection Details**
- Expected values for the forecast period (use exact projections provided)
- Confidence range: What's the optimistic vs pessimistic scenario?
- Key assumptions underlying this projection

**3. What's Driving the Trend**
- Is this organic growth or driven by specific changes?
- Are there seasonal factors at play?
- External factors to consider (competition, market conditions)

**4. Risk Assessment**
- What could cause this forecast to be wrong?
- Downside scenario: What happens if the negative factors materialize?
- Upside scenario: What could accelerate positive trends?

**5. Proactive Recommendations**
- What should be done NOW to capitalize on positive trends?
- What should be done to mitigate negative trends?
- Key metrics to monitor weekly

Use exact projection numbers provided. Format currency in ₹. Be forward-looking and strategic."""

    elif query_type == 'scenario':
        return f"""User Question: {user_question}

Analysis Type: Budget Scenario Simulation

Pre-Computed Analysis Results:
{data_json}

Provide a strategic scenario analysis:

**1. Scenario Overview**
- Clearly describe what change is being simulated
- Current state vs proposed state

**2. Projected Impact** (use exact numbers from data)
- Conversions: Current → Projected (% change)
- CPA: Current → Projected (% change)
- ROAS: Current → Projected (% change)
- Total Spend: Current → Projected

**3. Verdict: Should You Do This?**
- Clear YES/NO recommendation with reasoning
- Is the trade-off favorable?
- What's the risk level (Low/Medium/High)?

**4. Trade-offs Analysis**
- What do you gain? (Be specific)
- What do you lose or risk? (Be honest)
- Break-even point: When does this scenario make sense?

**5. Implementation Guidance** (if recommended)
- How to execute this change safely
- Recommended timeline (gradual vs immediate)
- Key metrics to monitor during transition
- Rollback triggers: When to reverse the change

**6. Alternative Scenarios to Consider**
- Would a smaller/larger change be better?
- Other levers that could achieve similar results

Use exact projected numbers. Format currency in ₹. Be decisive and practical."""

    elif query_type == 'recommendation':
        return f"""User Question: {user_question}

Analysis Type: Budget Reallocation Recommendation

Pre-Computed Analysis Results:
{data_json}

Provide strategic, actionable recommendations:

**1. Top Recommendation** (Lead with the verdict)
- State the single most impactful action to take
- Be specific: "Shift X% from Platform A to Platform B" not "Consider reallocation"

**2. Supporting Analysis**
- Why this recommendation? (efficiency gaps, performance trends)
- Quantify the opportunity cost of NOT doing this
- Current state vs optimal state

**3. Expected Impact** (use exact numbers)
- Projected change in conversions
- Projected change in CPA
- Projected change in ROAS
- Net benefit in ₹ terms

**4. Implementation Roadmap**
- Priority 1 (Do immediately): Highest impact, lowest risk actions
- Priority 2 (Do this week): Important optimizations
- Priority 3 (Monitor and plan): Longer-term strategic moves

**5. Risk Mitigation**
- What could go wrong with this recommendation?
- How to test safely before full commitment
- Warning signs to watch for

**6. What NOT to Do**
- Common mistakes to avoid
- Why certain "obvious" moves might backfire

Be decisive and specific. Use exact numbers. Format currency in ₹. Act like a senior media strategist advising a CMO."""

    elif query_type == 'lookup':
        return f"""User Question: {user_question}

Analysis Type: Data Lookup

Pre-Computed Analysis Results:
{data_json}

Provide a clear, informative answer:

**1. Direct Answer**
- State the specific number or data point requested
- Format currency in ₹ with Indian notation (L for lakhs, Cr for crores)

**2. Context** (brief, 1-2 sentences)
- How does this compare to prior period?
- Is this number good, average, or concerning?

**3. Quick Insight**
- One actionable observation related to this data point

Keep it concise but valuable. Don't just read numbers - add context."""

    else:  # general
        return f"""User Question: {user_question}

Pre-Computed Analysis Results:
{data_json}

Provide a helpful, strategic response:

**1. Direct Answer**
- Address the user's question clearly and directly
- Use exact numbers from the data provided

**2. Relevant Context**
- Additional insights that help understand the answer
- How does this relate to overall campaign performance?

**3. Suggested Next Questions**
- What follow-up questions might provide more value?
- What related analysis would be helpful?

Use ₹ for currency. Be professional and strategic. Only use numbers from the provided data."""


def build_conversational_prompt(
    query_type: str,
    user_question: str,
    analysis_data: dict,
    conversation_history: list = None
) -> str:
    """
    Build a prompt that includes conversation history for follow-up questions.

    Args:
        query_type: Type of analysis
        user_question: Current question
        analysis_data: Pre-computed analysis results
        conversation_history: List of previous Q&A pairs [{question, answer, analysis_summary}]
    """
    import json

    base_prompt = build_analysis_prompt(query_type, user_question, analysis_data)

    if not conversation_history:
        return base_prompt

    # Build conversation context
    history_text = "\n\n--- CONVERSATION HISTORY ---\n"
    for i, turn in enumerate(conversation_history[-5:], 1):  # Keep last 5 turns
        history_text += f"\n**Previous Question {i}:** {turn.get('question', '')}\n"
        history_text += f"**Previous Answer Summary:** {turn.get('answer_summary', turn.get('answer', '')[:500])}\n"

    history_text += "\n--- END CONVERSATION HISTORY ---\n\n"

    # Insert history before the analysis data
    return f"""This is a follow-up question in an ongoing conversation. Consider the previous context when responding.

{history_text}

**Current Question:** {user_question}

{base_prompt}

IMPORTANT: Since this is a follow-up question:
- Reference relevant points from the conversation history
- Don't repeat information already covered unless asked
- Build on previous insights
- If the question refers to "it", "that", "this", etc., understand the reference from context"""


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
