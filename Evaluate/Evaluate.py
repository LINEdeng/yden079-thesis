import os
import json
import argparse
from datasets import Dataset

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    context_entity_recall,
)

def build_models(backend: str, model_name: str):
    backend = backend.lower()
    print(f" Initializing backend: {backend}")

    if backend == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Please install it via `pip install langchain-openai`.")
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

        llm = ChatOpenAI(model=model_name, temperature=0)
        embed = OpenAIEmbeddings(model="text-embedding-3-large")
        print(f"Using OpenAI model: {model_name}")

    elif backend == "local":
        if ChatOllama is None:
            raise ImportError("langchain-ollama is not installed. Please install it via `pip install langchain-ollama`.")

        llm = ChatOllama(model=model_name, temperature=0)
        embed = OllamaEmbeddings(model=model_name)
        print(f"✅ Using local Ollama model: {model_name}")

    else:
        raise ValueError("Invalid backend. Use 'openai' or 'local'.")

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embed = LangchainEmbeddingsWrapper(embed)
    return wrapped_llm, wrapped_embed

def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation with GPT-4 or Local Ollama Model")
    parser.add_argument("--backend", type=str, default="local", choices=["openai", "local"], help="Choose backend")
    parser.add_argument("--model", type=str, default="llama3.1-8b-instruct:latest", help="Model name for backend")
    parser.add_argument("--file", type=str, default="rag_results.json", help="Path to results JSON file")
    parser.add_argument("--batch-size", type=int, default=6, help="Evaluation batch size")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ragas_data = []
    for item in data:
        if item.get("retrieved_context"):
            ragas_data.append({
                "question": item["question"],
                "contexts": [item["retrieved_context"]],
                "answer": item["generated_answer"],
                "ground_truth": item["gold_answer"],
            })

    ds = Dataset.from_list(ragas_data)
    print(f"Prepared dataset with {len(ds)} samples")

    wrapped_llm, wrapped_embed = build_models(args.backend, args.model)

    metrics_to_use = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
        context_entity_recall,
    ]

    # Run evaluation
    print("Running RAGAS evaluation ...")
    result = evaluate(
        dataset=ds,
        metrics=metrics_to_use,
        llm=wrapped_llm,
        embeddings=wrapped_embed,
        batch_size=args.batch_size,
    )

    print("Evaluation complete.")
    print(result)


if __name__ == "__main__":
    main()
