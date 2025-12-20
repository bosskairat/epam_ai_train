import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from transformers import pipeline


# Load environment variables from .env file
load_dotenv()

HF_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# Text generation model for local run.
LOCAL_LLM_MODEL_NAME = "google/gemma-3-1b-it"

# Login to Hugging Face Hub
login(token=HF_API_TOKEN)
print("Successfully logged in to Hugging Face!")


# --- Wrapper class for the local LLM ---
class LocalHuggingFaceChatModel(Runnable):
    """
    A simple wrapper around the Transformers Pipeline to make it compatible
    with LangChain's 'invoke' method and the pipe '|' operator.
    """
    def __init__(self, model_name=LOCAL_LLM_MODEL_NAME):
        print(f"📥 Loading local LLM: {model_name}...")
        # This is the 'Automatic Transmission' setup we discussed:
        # 1. device=-1 forces CPU usage.
        # 2. torch_dtype=torch.float32 is the fastest format for CPU.
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )
        print("✅ Local LLM loaded successfully.")

    def invoke(self, input_data, config: RunnableConfig = None, **kwargs):
        """
        Adapts LangChain inputs (PromptValue or Messages) to the pipeline format.
        """
        # 1. Convert LangChain input to the list-of-dicts format expected by the pipeline
        messages = []

        # Handle LangChain PromptValue (which has .to_messages())
        if hasattr(input_data, 'to_messages'):
            lc_messages = input_data.to_messages()
            for msg in lc_messages:
                # Map LangChain message types to role strings
                role = "user"
                if msg.type == "system": role = "system"
                elif msg.type == "ai": role = "assistant"

                # Gemma pipeline expects content as a list of dicts or string.
                messages.append({"role": role, "content": [{"type": "text", "text": msg.content}]})

        # Handle raw string input (fallback)
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": input_data}]}]

        # 2. Run the pipeline ("Automatic Transmission")
        # We set max_new_tokens to limit the answer length
        outputs = self.pipe(messages, max_new_tokens=512)

        # 3. Extract the generated text
        # The pipeline returns a list of dicts. The last message is the assistant's reply.
        generated_text = outputs[0]['generated_text'][-1]['content']

        # 4. Return as an AIMessage to satisfy LangChain's StrOutputParser
        return AIMessage(content=generated_text)