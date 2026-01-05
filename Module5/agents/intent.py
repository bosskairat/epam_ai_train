import json
from agents.llm import OpenAILLM

llm = OpenAILLM()

SYSTEM_PROMPT = """
You are an intent parser. Extract the user's request about weather and news from their query. 

Return STRICT JSON only, exactly in this format:

{
  "weather": boolean,
  "news": boolean,
  "city": string | null,
  "date": "today" | "tomorrow" | "next N days" | null,
  "news_topic": string | null
}

- "weather": true if the query asks about weather
- "news": true if the query asks about news
- "city": the city name mentioned in the query, or null if not mentioned
- "date": 
    - "today" if the query mentions today
    - "tomorrow" if the query mentions tomorrow
    - "next N days" if the query mentions a multi-day forecast (e.g., next 3 days)
    - null if not specified
- "news_topic": topic mentioned for news, or null if none

Do not include any text outside the JSON. Only return JSON. 

Query: "{user_query}"
"""

def parse_intent(user_query: str) -> dict:
    result = llm.chat(SYSTEM_PROMPT, user_query)
    print(result)

    try:
        return json.loads(result)
    except Exception:
        return {
            "weather": False,
            "news": False,
            "city": None,
            "date": None,
            "news_topic": None
        }
