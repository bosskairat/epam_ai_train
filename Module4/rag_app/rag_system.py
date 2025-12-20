import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from llm import LocalHuggingFaceChatModel
from embeddings import LocalHuggingFaceEmbeddings


# Load environment variables from .env file
load_dotenv() 

HF_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

WEAVIATE_HTTP_PORT_EXTERNAL = os.environ["WEAVIATE_HTTP_PORT_EXTERNAL"]
WEAVIATE_GRPC_PORT_EXTERNAL = os.environ["WEAVIATE_GRPC_PORT_EXTERNAL"]
COLLECTION_NAME = os.environ["WEAVIATE_COLLECTION_NAME"]


class RAG:

    def __init__(self):

        # Re-connect to Weaviate for the experiment
        self.weaviate_client = weaviate.connect_to_local(
            host="localhost",
            port=WEAVIATE_HTTP_PORT_EXTERNAL,
            grpc_port=WEAVIATE_GRPC_PORT_EXTERNAL
        )
    
        print("--- Setting up AI clients ---")
        try:
            # Embedding Model Setup
            self.embeddings_model = LocalHuggingFaceEmbeddings()
            # Chat Model Setup
            self.chat_model = LocalHuggingFaceChatModel()
            print("✅ AI clients initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize AI clients. Please check your .env file or model names. Error: {e}")
            # Stop execution if clients fail to initialize
            raise


    def data_ingestion(self, documents_data):

        contents_to_embed = [doc['content'] for doc in documents_data]
        vector_embeddings = self.embeddings_model.embed_documents(contents_to_embed)
        print(f"✅ Generated {len(vector_embeddings)} embeddings. Vector dimension: {len(vector_embeddings[0])}")
        # Add embeddings to our data
        for i, doc in enumerate(documents_data):
            doc['content_vector'] = vector_embeddings[i]

        # Delete collection if it already exists for a clean run
        if self.weaviate_client.collections.exists(COLLECTION_NAME):
            self.weaviate_client.collections.delete(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")

        # Create new DB schema for our documents
        rag_collection = self.weaviate_client.collections.create(
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



    def answer_the_question(self, user_query: str):
        # 1. Chain for Query Expansion
        expansion_prompt = ChatPromptTemplate.from_template(
            "You are an expert in information retrieval. "
            "Please rephrase the following user query to be more descriptive and detailed, "
            "making it suitable for a vector database search. "
            "Return only the rephrased query, without any additional text, headers, or explanations. "
            "\n\nOriginal Query: '{query}'\n\nRephrased Query:"
        )
        query_expansion_chain = expansion_prompt | self.chat_model | StrOutputParser()

        # 2. Chain for Final Answer Generation (with RAG context)
        # generation_prompt = ChatPromptTemplate.from_template(
        #     "You are a helper working with an article review. "
        #     "Your task is to answer the user's question based only on the provided context, "
        #     "do not use common knowledge, do not correct mistakes in provided context. "
        #     "Synthesize the information from the context into a detailed summary. "
        #     "Focus on specific details like names, numbers, and technical terms mentioned in the context. "
        #     "If the context does not contain the information needed to answer the question, "
        #     "you must state: 'The provided context does not contain the answer to this question.' "
        #     "\n\nContext:\n{context}\n\nQuestion: {question}"
        # )
        # generation_prompt = ChatPromptTemplate.from_template(
        #     "You are a article reviewer. "
        #     "Your task is to answer the user's question based only on the provided context, "
        #     "do not use common knowledge, do not correct mistakes in provided context. "
        #     "Synthesize the detailed information from the context. " 
        #     "If the context does not contain the information needed to answer the question, "
        #     "you must state: 'The provided context does not contain the answer to this question.' "
        #     "\n\nContext:\n{context}\n\nQuestion: {question}"
        # )
        generation_prompt = ChatPromptTemplate.from_template(
            "You are a factual assistant. "
            "Your task is to answer the user's question based only on the provided context, "
            "do not use common knowledge, do not correct mistakes in provided context. "
            "Synthesize the information from the context into a concise, bullet-point summary. "
            "Focus on specific details like names, numbers, and technical terms mentioned in the context. "
            "If the context does not contain the information needed to answer the question, "
            "you must state: 'The provided context does not contain the answer to this question.' "
            "\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        answer_generation_chain = generation_prompt | self.chat_model | StrOutputParser()

        # --- Run the Experiment ---
        print(f"### Original Question: {user_query}")

        # 1. Expand the query
        expanded_query = query_expansion_chain.invoke({"query": user_query})
        print(f"**Rephrased Query for Search:** {expanded_query}")

        # --- RAG Pipeline ---

        # Embed Expanded Query
        query_embedding = self.embeddings_model.embed_query(expanded_query)

        # Retrieve Documents from Weaviate
        rag_collection = self.weaviate_client.collections.get(COLLECTION_NAME)
        retrieved_objects = rag_collection.query.near_vector(
            near_vector=query_embedding,
            limit=10,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        retrieved_docs_content = [obj.properties['content'] for obj in retrieved_objects.objects]
        
        context_for_llm = "\n\n---\n\n".join(retrieved_docs_content)

        # 3. Generate Final Answer using RAG
        final_answer = answer_generation_chain.invoke({
            "context": context_for_llm,
            "question": user_query
        })
        print(f"**Answer to the original query, with RAG:**\n{final_answer}")

        retrieved_docs_content_with_ref =  [f"{obj.properties['content']} [{obj.properties['file']}]" for obj in retrieved_objects.objects]
        context_for_show = "\n\n---\n\n".join(retrieved_docs_content_with_ref)

        return expanded_query, context_for_show, final_answer
    

    def close_weaviate_client(self):
        # Close the client connection
        self.weaviate_client.close()


if __name__ == "__main__":
    rag_system = RAG()
    rag_system.answer_the_question("What methods were used in the article?")
    rag_system.close_weaviate_client()