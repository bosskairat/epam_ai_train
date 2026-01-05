import json
from agents.llm import OpenAILLM

llm = OpenAILLM()

# Added {history} placeholder to the prompt
SYSTEM_PROMPT = """
You are an intent parser. Extract the user's request about weather and news from their query. 
Use the provided conversation history to resolve references like "there", "it", or "tomorrow".

Return STRICT JSON only, exactly in this format:

{{
  "weather": boolean,
  "news": boolean,
  "city": string | null,
  "date": "today" | "tomorrow" | "next N days" | null,
  "news_topic": string | null
}}

- "weather": true if the query asks about weather
- "news": true if the query asks about news
- "city": the city name mentioned in the query or inferred from history, or null
- "date": 
    - "today" if the query mentions today
    - "tomorrow" if the query mentions tomorrow
    - "next N days" if the query mentions a multi-day forecast
    - null if not specified
- "news_topic": topic mentioned for news or inferred from history, or null

Conversation History:
{history}

Current Query: "{user_query}"
"""

def parse_intent(user_query: str, history_text: str) -> dict:
    # We format the prompt with both history and the current query
    formatted_prompt = SYSTEM_PROMPT.format(history=history_text, user_query=user_query)
    
    # Note: Depending on your OpenAILLM implementation, 
    # you might pass the formatted prompt as the system message.
    result = llm.chat(formatted_prompt, user_query)

    try:
        # Clean up potential markdown formatting from LLM (like ```json ... ```)
        json_str = result.strip().replace("```json", "").replace("```", "")
        return json.loads(json_str)
    except Exception:
        return {
            "weather": False,
            "news": False,
            "city": None,
            "date": None,
            "news_topic": None
        }