import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import weaviate
import weaviate.classes as wvc
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
        self.rag_collection = self.weaviate_client.collections.get(COLLECTION_NAME)

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
        generation_prompt = ChatPromptTemplate.from_template(
            "You are a helper working with an article review. "
            "Your task is to answer the user's question based only on the provided context, "
            "do not use common knowledge, do not correct mistakes in provided context. "
            "Synthesize the information from the context into a detailed summary. "
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
        retrieved_objects = self.rag_collection.query.near_vector(
            near_vector=query_embedding,
            limit=5,
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

        return final_answer
    

    def close_weaviate_client(self):
        # Close the client connection
        self.weaviate_client.close()


if __name__ == "__main__":
    rag_system = RAG()
    rag_system.answer_the_question("What methods were used in the article?")
    rag_system.close_weaviate_client()