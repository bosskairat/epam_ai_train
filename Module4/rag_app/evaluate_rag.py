import time
import json
from datetime import datetime
from rag_system import RAG # Assuming your code is in rag_system.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class RAGEvaluator:
    def __init__(self, rag_instance):
        self.rag = rag_instance
        # Example test set: (Query, Expected Key Fact)
        self.test_set = [
            {"query": "Which face recognition model and comparison method achieved the best performance according to the paper?",
             "expected_fact": "The best-performing approach was the Dlib face recognition model combined with an SVM classifier, \
                which achieved the highest ability to distinguish individuals among the evaluated combinations."},
            {"query": "Explain the meaning of [specific term] mentioned in the articles."},
            {"query": "What numbers, statistics, or results are reported in the articles?"}
        ]

    def evaluate_faithfulness(self, answer, context):
        """
        Evaluates the credibility of a response based on the provided context.
        Uses the NLI (Natural Language Inference) method.
        """
        
        # 1. If the model recognizes the absence of data, this is correct behavior (Score 1.0)
        if "context does not contain" in answer.lower():
            return 1.0

        # 2. Prompt for checking assertions
        verification_prompt = ChatPromptTemplate.from_template(
            "You are an auditor. Your task is to check if the 'Answer' is fully supported by the 'Context'.\n"
            "Break the answer into logical claims and check each one.\n"
            "Output ONLY a score between 0.0 and 1.0, where:\n"
            "1.0: Every claim in the answer is explicitly supported by the context.\n"
            "0.0: The answer contains information not found in the context (hallucination).\n"
            "--- \n"
            "Context: {context}\n"
            "Answer: {answer}\n"
            "--- \n"
            "Score (number only):"
        )

        verification_chain = verification_prompt | self.rag.chat_model | StrOutputParser()

        try:
            # 3. Launching the assessment
            response = verification_chain.invoke({
                "context": context,
                "answer": answer
            })
            
            # Extract the number from the response (in case the model adds text)
            score_match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", response)
            if score_match:
                return float(score_match.group(1))
            return 0.5 # Default if parsing failed
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0


    def run_suite(self):
        results = []
        print(f"--- Starting Evaluation Suite: {datetime.now()} ---")

        for test in self.test_set:
            start_time = time.time()
            
            # Run the actual RAG pipeline
            expanded_q, context, answer = self.rag.answer_the_question(test['query'])
            
            latency = time.time() - start_time
            
            # Calculate Metrics
            faithfulness = self.evaluate_faithfulness(answer, context)
            
            # Log results
            results.append({
                "query": test['query'],
                "latency": round(latency, 2),
                "faithfulness": faithfulness,
                "context_len": len(context)
            })

        self.generate_report(results)

    def generate_report(self, results):
        avg_latency = sum(r['latency'] for r in results) / len(results)
        avg_faith = sum(r['faithfulness'] for r in results) / len(results)
        
        report = f"""
        # RAG SYSTEM PERFORMANCE REPORT
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        -----------------------------------------
        Total Tests: {len(results)}
        Avg Latency: {avg_latency:.2f} s
        Avg Faithfulness: {avg_faith:.2f}
        
        Details:
        {json.dumps(results, indent=2)}
        """
        with open("rag_report.txt", "w") as f:
            f.write(report)
        print("✅ Report generated: rag_report.txt")


if __name__ == "__main__":
    rag = RAG()
    evaluator = RAGEvaluator(rag)
    evaluator.run_suite()
    rag.close_weaviate_client()