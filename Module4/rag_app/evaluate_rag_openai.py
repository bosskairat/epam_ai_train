import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from rag_system import RAG


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

        Score the model answer from 0 to 5 based on:
        1. Factual correctness
        2. Completeness
        3. Faithfulness to the source

        Return ONLY a JSON object:
        {{"score": <0-5>, "justification": "<short explanation>"}}
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def evaluate_rag(dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)

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
    scores = evaluate_rag("data/test/59_2021_Qualitative Evaluation of Face Embeddings.json")
    my_rag.close_weaviate_client()
    avg_score = sum(r["score"] for r in scores) / len(scores)
    print(f"Average RAG Score: {avg_score:.2f}")
    with open("rag_evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=2)