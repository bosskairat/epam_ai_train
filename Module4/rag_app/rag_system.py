import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from llm import LocalHuggingFaceChatModel
from embeddings import LocalHuggingFaceEmbeddings
from reranker import Reranker
from hyde import HyDE


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
            # HyDE generator (uses the same chat model by default)
            self.hyde = HyDE(llm=self.chat_model, include_original=False)
            # Reranker (cross-encoder with fallback)
            try:
                self.reranker = Reranker()
            except Exception:
                self.reranker = None
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
        # Baseline prompt
        # generation_prompt = ChatPromptTemplate.from_template(
        #     "You are a factual assistant. "
        #     "Your task is to answer the user's question based only on the provided context, "
        #     "do not use common knowledge, do not correct mistakes in provided context. "
        #     "Synthesize the information from the context into a concise, bullet-point summary. "
        #     "Focus on specific details like names, numbers, and technical terms mentioned in the context. "
        #     "If the context does not contain the information needed to answer the question, "
        #     "you must state: 'The provided context does not contain the answer to this question.' "
        #     "\n\nContext:\n{context}\n\nQuestion: {question}"
        # )
        # Improved prompt
        generation_prompt = ChatPromptTemplate.from_template(
            "SYSTEM: You are an expert factual analyst. Your sole purpose is to answer the "
            "user's question using only the provided context. \n\n"
            
            "RULES:\n"
            "1. Only use the information provided in the Context. Never use external knowledge.\n"
            "2. If the Context is insufficient, state exactly: 'The provided context does not contain the answer.'\n"
            "3. Do not correct typos or factual errors within the Context; report them as written.\n"
            "4. Prioritize technical terms, specific dates, names, and numerical data.\n"
            "5. Provide a structured response: Start with a direct answer, followed by supporting bullet points.\n\n"
            
            "CONTEXT:\n{context}\n\n"
            
            "USER QUESTION: {question}\n\n"
            "ASSISTANT ANSWER:"
        )

        answer_generation_chain = generation_prompt | self.chat_model | StrOutputParser()

        # --- Run the Experiment ---
        # print(f"### Original Question: {user_query}")

        # 1. Expand the query
        expanded_query = query_expansion_chain.invoke({"query": user_query})
        # print(f"**Rephrased Query for Search:** {expanded_query}")

        # Improve retrieval with HyDE
        # HyDE: generate a hypothetical document from the (expanded) query
        try:
            hyde_doc = None
            if hasattr(self, 'hyde') and self.hyde is not None:
                expanded_query = self.hyde.transform(expanded_query)
        except Exception as e:
            print(f"⚠️ HyDE generation failed: {e}")

        # --- RAG Pipeline ---

        # Embed Expanded Query
        query_embedding = self.embeddings_model.embed_query(expanded_query)

        # Retrieve Documents from Weaviate
        rag_collection = self.weaviate_client.collections.get(COLLECTION_NAME)
        retrieved_objects = rag_collection.query.near_vector(
            near_vector=query_embedding,
            limit=15,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        retrieved_objects_list = list(retrieved_objects.objects)
        retrieved_docs_content = [obj.properties['content'] for obj in retrieved_objects_list]

        # Improve retrieval with Reranker
        # Rerank retrieved documents if reranker is available
        try:
            if hasattr(self, 'reranker') and self.reranker is not None:
                ranked = self.reranker.rerank(expanded_query, retrieved_docs_content, top_k=10)
                if ranked:
                    retrieved_docs_content = [r['doc'] for r in ranked]
        except Exception as e:
            print(f"⚠️  Reranking in RAG failed: {e}")

        context_for_llm = "\n\n---\n\n".join(retrieved_docs_content)

        # 3. Generate Final Answer using RAG
        final_answer = answer_generation_chain.invoke({
            "context": context_for_llm,
            "question": user_query
        })
        # print(f"**Answer to the original query, with RAG:**\n{final_answer}")

        retrieved_docs_content_with_ref = [f"{obj.properties['content']} [{obj.properties['file']}]" for obj in retrieved_objects.objects]
        context_for_show = "\n\n---\n\n".join(retrieved_docs_content_with_ref)

        return expanded_query, context_for_show, final_answer
    

    def close_weaviate_client(self):
        # Close the client connection
        self.weaviate_client.close()


if __name__ == "__main__":
    rag_system = RAG()
    rag_system.answer_the_question("What methods were used in the article?")
    rag_system.close_weaviate_client()