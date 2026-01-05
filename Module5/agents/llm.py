from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


class OpenAILLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
