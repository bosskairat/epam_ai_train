import os
import csv
from dotenv import load_dotenv
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from embeddings import LocalHuggingFaceEmbeddings


# --- 0. Load environment variables from .env file
load_dotenv()
WEAVIATE_HTTP_PORT_EXTERNAL = os.environ["WEAVIATE_HTTP_PORT_EXTERNAL"]
WEAVIATE_GRPC_PORT_EXTERNAL = os.environ["WEAVIATE_GRPC_PORT_EXTERNAL"]
COLLECTION_NAME = os.environ["WEAVIATE_COLLECTION_NAME"]


# --- 1. Load document chunks ---
print("--- 1. Loading document chunks ---")
document_path = os.path.join(os.path.dirname(__file__), "data/article_chunks.csv")
with open(document_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    documents_data = list(reader)

# --- 2. Setup LangChain Clients ---
print("--- 2. Setting up AI clients ---")
try:
    # Embedding Model Setup
    embeddings_model = LocalHuggingFaceEmbeddings()
    print("✅ AI clients initialized.")
except Exception as e:
    print(f"❌ Failed to initialize AI clients. Please check your .env file or model names. Error: {e}")
    # Stop execution if clients fail to initialize
    raise

# --- 3. Generate Embeddings ---
print("\n--- 3. Generating embeddings for all documents ---")
contents_to_embed = [doc['content'] for doc in documents_data]
vector_embeddings = embeddings_model.embed_documents(contents_to_embed)
print(f"✅ Generated {len(vector_embeddings)} embeddings. Vector dimension: {len(vector_embeddings[0])}")

# Add embeddings to our data
for i, doc in enumerate(documents_data):
    doc['content_vector'] = vector_embeddings[i]

# --- 3. Connect to Weaviate ---
print("\n--- 3. Connecting to Weaviate ---")
weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=WEAVIATE_HTTP_PORT_EXTERNAL,
    grpc_port=WEAVIATE_GRPC_PORT_EXTERNAL
)
if weaviate_client.is_ready():
    print("✅ Successfully connected to Weaviate.")
else:
    print("❌ Failed to connect to Weaviate.")
    weaviate_client.close()
    raise ConnectionError("Could not connect to Weaviate instance.")

# --- 4. Define and Create Weaviate Collection ---

print(f"\n--- 4. Creating Weaviate collection: '{COLLECTION_NAME}' ---")

# Delete collection if it already exists for a clean run
if weaviate_client.collections.exists(COLLECTION_NAME):
    weaviate_client.collections.delete(COLLECTION_NAME)
    print(f"Deleted existing collection '{COLLECTION_NAME}'.")

# Create new DB schema for our documents
rag_collection = weaviate_client.collections.create(
    name=COLLECTION_NAME,
    properties=[
        wvc.config.Property(name="file", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ],
    vector_config=wvc.config.Configure.Vectors.self_provided(
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE
        )
    )
)
print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")


# --- 5. Batch-Insert Data ---
print(f"\n--- 5. Ingesting {len(documents_data)} documents into Weaviate ---")
# Use a context manager to automatically handle batching
with rag_collection.batch.dynamic() as batch:
    for doc in documents_data:
        properties = {
            "file": doc["file"],
            "chunk_id": doc["chunk_id"],
            "content": doc["content"]
        }
        batch.add_object(
            properties=properties,
            vector=doc["content_vector"],  # Use default vector
            uuid=generate_uuid5(doc["file"] + "_" + doc["chunk_id"])  # Generate a consistent UUID based on the file and chunk_id
        )
print(f"✅ Data ingestion complete. Total objects in collection: {len(rag_collection)}")

# Close the client connection
weaviate_client.close()
