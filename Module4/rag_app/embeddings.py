import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer


# Load environment variables from .env file
load_dotenv()

HF_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

WEAVIATE_HTTP_PORT_EXTERNAL = os.environ["WEAVIATE_HTTP_PORT_EXTERNAL"]
WEAVIATE_GRPC_PORT_EXTERNAL = os.environ["WEAVIATE_GRPC_PORT_EXTERNAL"]
COLLECTION_NAME = os.environ["WEAVIATE_COLLECTION_NAME"]

# Embeddingd model for local run.
LOCAL_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

# Login to Hugging Face Hub
login(token=HF_API_TOKEN)
print("Successfully logged in to Hugging Face!")


# --- Wrapper class for the local Embeddings model ---
class LocalHuggingFaceEmbeddings:
    """
    This class adapts a local SentenceTransformer model
    to the LangChain interface, which expects the methods embed_documents and embed_query.
    """
    def __init__(self, model_name=LOCAL_EMBEDDING_MODEL_NAME):
        print(f"📥 Loading local embedding model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Local embedding model loaded successfully. Device: " + ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception as e:
            print(f"❌ Error loading {model_name}. Falling back to 'all-MiniLM-L6-v2'.")
            print(f"Error details: {e}")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        # Returns a list of lists
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        # Returns a single list
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()