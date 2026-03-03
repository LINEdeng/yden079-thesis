# AdaIterDAR (DARAG)

**Adaptive Multi-Round RAG for Complex Multi-Hop Question Answering**

针对传统 RAG 在复杂多跳问答任务中普遍存在的信息冗余与证据覆盖不足问题，本项目实现了一种自适应多轮 RAG 方法 **DARAG**（AdaIterDAR），通过动态调度检索与生成、结合 MMR 重排序，在保证响应效率的前提下提升证据覆盖率与多样性，增强多跳推理链条的完整性与信息充分性。

---

## Overview

Traditional RAG methods often suffer from **information redundancy** and **insufficient evidence coverage** on complex multi-hop QA tasks. We propose **AdaIterDAR** (DARAG), an adaptive multi-round RAG approach that:

- **Dynamically schedules** retrieval and generation: single-round for simple questions, multi-round iterative retrieval and reasoning for complex ones.
- **Introduces MMR (Maximal Marginal Relevance)** in the reranking stage to improve **diversity** and **evidence coverage**, reducing redundant content while keeping high relevance.

### Main Results (HotpotQA)

| Metric | Improvement |
|--------|-------------|
| Answer Correctness | **+8%** |
| Context Precision | **+6.5%** |
| Context Recall | **+8%** |

Retrieval results are more diverse and provide higher evidence coverage, offering more reliable support for multi-hop QA.

---

## Core Capabilities

### 1. Question Complexity Awareness & Multi-Round Scheduling

- A **question complexity classifier** is placed in front of the retriever to route questions by type.
- **Simple (single-hop)** questions: one-round retrieval and answer for faster response.
- **Complex (multi-hop)** questions: multi-round iterative retrieval and reasoning (decomposition → coreference resolution → per-subquestion retrieval → integration).

### 2. MMR-Based Reranking for Diversity

- **Maximal Marginal Relevance (MMR)** is used in the reranking stage.
- In each retrieval round, candidate documents are **deduplicated** and **diversified**, so that selected passages are both relevant and **complementary** in content, mitigating redundancy and evidence gaps caused by pure relevance ranking.

---

## Project Structure

```
AdaIterDAR/
├── approach/
│   └── DARAG.py          # Main pipeline: classifier, retriever, MMR rerank, multi-hop answerer
├── classifier/           # Question complexity classifier (train & inference)
│   ├── preprocess/
│   ├── run/
│   └── postprocess/
├── data/
│   ├── load_hotpotqa.py  # HotpotQA data loading
│   ├── load_msmarco.py
│   ├── load_pubmedqa.py
│   └── dataloder.py
├── Evaluate/
│   └── Evaluate.py       # RAGAS evaluation (correctness, precision, recall, etc.)
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch
- [LangChain](https://github.com/langchain-ai/langchain) (HuggingFace embeddings, Chroma, Ollama)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) (reranker)
- [transformers](https://github.com/huggingface/transformers) (T5 for classifier, decomposer, coref)
- [ragas](https://github.com/explodinggradients/ragas) (for evaluation)

---

## Data & Models

- **Dataset**: HotpotQA (distractor setting). Use `data/load_hotpotqa.py` to download or load from Hugging Face.
- **Vector DB**: Build a Chroma index from your corpus (e.g. HotpotQA contexts) using the same embedding model as in `approach/DARAG.py` (e.g. BGE).
- **Pre-trained components** (paths are configurable in code):
  - Question classifier (T5)
  - Question decomposer (T5)
  - Coreference resolver (T5)
  - Embedding: e.g. BGE-base-en-v1.5
  - Reranker: e.g. BGE-reranker-large
- **LLM**: Local Ollama (e.g. `llama3.1-8b-instruct`) or replace with another LangChain-compatible backend.

---

## Usage

### 1. Build vector store and run inference

In `approach/DARAG.py`, set:

- `model_path`: embedding model path  
- `db_path`: Chroma DB path  
- `question_file`: path to QA JSON (e.g. `../data/hotpot_qa.json`)

Then run:

```bash
cd approach
python DARAG.py
```

Output: `rag_results.json` with `question`, `retrieved_context`, `generated_answer`, `gold_answer` per sample.

### 2. Evaluate with RAGAS

```bash
cd Evaluate
python Evaluate.py --backend local --model llama3.1-8b-instruct:latest --file ../approach/rag_results.json
```

Options: `--backend` (`local` | `openai`), `--model`, `--file`, `--batch-size`.

---

## Pipeline Summary

1. **Access**: Input question → **Question Classifier** (A/B/C: simple / medium / complex).
2. **Attempt**: For simple/medium (A/B), single-round: **DAR Retriever** → dedup → **MMR rerank** → LLM. If the answer is *"I don't know"*, fall back to multi-hop.
3. **Decompose**: For complex (C) or fallback: **Question Decomposition** → **Coreference Resolver** → rewritten sub-questions → per-subquestion **DAR** + MMR → sub-answers.
4. **Integrate**: Combine sub-questions, sub-answers, and deduplicated context → final LLM call → **Candidate answer** → output.

---

## Citation

If you use this code or idea in your work, please cite:

```bibtex
@misc{adaiterdar,
  title={AdaIterDAR: Adaptive Multi-Round RAG for Multi-Hop Question Answering},
  author={...},
  year={2025},
  url={https://github.com/.../AdaIterDAR}
}
```

---

## License

See repository license file.

