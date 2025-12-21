Evaluation Script

The RAG evaluation script implements an automated LLM-as-a-judge approach to assess answer quality using a percentage-based scoring system (0–100%). For each question, the system compares the model-generated answer against a curated expected answer and evaluates it along three core dimensions: factual correctness, completeness, and faithfulness to the retrieved context. The evaluator LLM returns both a numeric score and a brief justification, enabling quantitative benchmarking as well as qualitative error analysis. This setup supports regression testing, comparison across retriever or prompt variants, and integration into CI pipelines, making it suitable for continuous RAG system improvement.

0. Baseline overall average score is 49.0


1. Improved splitting method with sentence-based chunks and overlap

Change: Replaced paragraph splitting with sentence-based sliding-window chunking and added overlap_sentences parameter.
How it works: Split into sentences, aggregate until max_block_size, ensure min_block_size when possible, then advance window with overlap to produce overlapping chunks.
Benefits: Improves passage coherence and recall for RAG, reduces fragmentary answers, and lets you tune overlap to boost faithfulness.
Defaults & recommendation: min_block_size 200–400 chars, max_block_size 800–1500 chars, overlap_sentences 1–3.
Next steps: Run quick experiments on a few PDFs, compare retrieval accuracy and index size, then consider switching to token-based sizing for tighter LLM context control.
cleaner = PDFCleaner(min_block_size=300, max_block_size=1000, overlap_sentences=2)
Improved splitting method overall average score is 52.6


2. Changed Prompt

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
The updated prompt improved the evaluation results, raising the overall average score to 58.30%.


3. The retrieval limit was reduced from 10 to 5:
retrieved_objects = rag_collection.query.near_vector( near_vector=query_embedding, limit=5, return_metadata=wvc.query.MetadataQuery(distance=True) )
However, this change did not improve the overall performance. The average evaluation score decreased to 44.40%, indicating reduced answer coverage despite higher precision.


4. Reranker Integration 

A reranking stage was added to the RAG pipeline to improve the relevance of retrieved documents before answer generation. After the initial vector-based retrieval, the retrieved passages are reordered using a reranker that prioritizes semantic relevance to the expanded query. The system primarily employs a cross-encoder reranker (ms-marco-MiniLM-L-6-v2), which jointly encodes the query–document pairs to produce more accurate relevance scores. To ensure robustness, a fallback mechanism based on embedding cosine similarity is used when the cross-encoder is unavailable. Only the top-ranked documents (top-10) are passed to the generation stage, reducing contextual noise and improving answer grounding. This reranking step enhances retrieval precision and contributes to more faithful and focused RAG outputs.
Overall average score is 48.70

5.Hypothetical Document Embeddings

 DEFAULT_PROMPT = (
        "Given the user question below, write a short hypothetical document "
        "that contains plausible facts and context that would help answer the question.\n\n"
        "Constraints:\n"
        "- Write ONLY 2 to 5 complete sentences.\n"
        "- Do NOT add explanations, lists, or extra commentary.\n"
        "- Do NOT mention that this is hypothetical.\n\n"
        "Question: {query}\n\n"
        "Hypothetical document:"
    )
verall average score is 51.00

6. Hypothetical Document Embeddings
self.hyde = HyDE(llm=self.chat_model, include_original=False)
Write ONLY 3 to 5 complete sentences