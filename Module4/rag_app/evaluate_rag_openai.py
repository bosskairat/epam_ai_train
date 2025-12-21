import os
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI
from rag_system import RAG
from pdf_loader import PDFCleaner


# Load environment variables from .env file
load_dotenv() 
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
my_rag = RAG()

def llm_judge(question, expected, predicted):
    prompt = f"""
        You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

        Question:
        {question}

        Expected Answer:
        {expected}

        Model Answer:
        {predicted}

        Score the model answer from 0 to 100 based on:
        1. Factual correctness
        2. Completeness
        3. Faithfulness to the source

        Return ONLY a JSON object:
        {{"score": <0-100>, "justification": "<short explanation>"}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def evaluate_rag(dataset_path):
    pdf_path = os.path.join("data/pdf", os.path.basename(dataset_path).replace(".json", ".pdf"))
    # Baseline splitting method 
    # cleaner = PDFCleaner(min_block_size=100, max_block_size=500)
    # Splitting method with sentence-based chunks and overlap
    cleaner = PDFCleaner(min_block_size=300, max_block_size=1000, overlap_sentences=1)
    chunks = cleaner.process_pdf(str(pdf_path))

    # collect documents to ingest into RAG
    documents_data =[
        {"file": os.path.basename(pdf_path), "chunk_id": str(i + 1), "content": chunk}
        for i, chunk in enumerate(chunks)
    ]
    my_rag.data_ingestion(documents_data)

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Evaluate each question in the dataset
    results = []
    for item in dataset:
        _, _, predicted = my_rag.answer_the_question(item["question"])
        judge = llm_judge(
            item["question"],
            item["expected_answer"],
            predicted
        )

        results.append({
            "id": item["id"],
            "question": item["question"],
            "predicted_answer": predicted,
            "expected_answer": item["expected_answer"],
            "score": judge["score"],
            "justification": judge["justification"]
        })

    return results


if __name__ == "__main__":
    all_datasets_output = []
    all_scores = []
    for dataset_path in sorted(glob.glob("data/test/*.json")):
        print(f"Evaluating dataset: {dataset_path}")
        scores = evaluate_rag(dataset_path)
        avg_score = sum(r["score"] for r in scores) / len(scores) if scores else 0.0
        print(f"Average RAG Score: {avg_score:.2f}")
        output = {
            "dataset": os.path.basename(dataset_path),
            "average_score": avg_score,
            "results": scores,
        }
        all_datasets_output.append(output)
        all_scores.extend([r["score"] for r in scores])

    overall_average = sum(all_scores) / len(all_scores) if all_scores else 0.0
    final_output = {
        "datasets": all_datasets_output,
        "overall_average_score": overall_average,
    }
    print(f"Overall Average RAG Score (all datasets): {overall_average:.2f}")
    with open("eval/4_rag_evaluation_results_hyde2.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    # Close the Weaviate client once after all datasets processed
    my_rag.close_weaviate_client()