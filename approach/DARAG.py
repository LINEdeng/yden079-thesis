import os
import numpy as np
import torch
import torch.optim as optim
import time
import json
from datasets import Dataset
from tqdm import tqdm
import concurrent.futures
import threading

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from FlagEmbedding import FlagReranker
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Load models
local_llm = "llama3.1-8b-instruct:latest"
scorer = ChatOllama(model=local_llm, temperature=0.7)
llm = ChatOllama(model=local_llm, temperature=0)
classifier_tokenizer = T5Tokenizer.from_pretrained("/home/yden079/.cache/train/models--question_classifier--t5/snapshots/sg89059b7aawuqi10482cd6664543f72a786fc6")
classifier_model = T5ForConditionalGeneration.from_pretrained("/home/yden079/.cache/train/models--question_classifier--t5/snapshots/sg89059b7aawuqi10482cd6664543f72a786fc6", torch_dtype=torch.float16)
similarity_model = SentenceTransformer("/home/yden079/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
rerank_model = FlagReranker("/home/yden079/.cache/huggingface/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312", use_fp16=True)
decomposer_tokenizer = T5Tokenizer.from_pretrained("/home/yden079/.cache/huggingface/hub/models--question_decomposer--t5/snapshots/e26e059b7aa4b9aef0482cd6664543f72a786fc6")
decomposer_model = T5ForConditionalGeneration.from_pretrained("/home/yden079/.cache/huggingface/hub/models--question_decomposer--t5/snapshots/e26e059b7aa4b9aef0482cd6664543f72a786fc6")
Coref_tokenizer = T5Tokenizer.from_pretrained("/home/yden079/.cache/huggingface/hub/models--coref--t5/snapshots/f0f21fc4cae5dc130d97e4fa4dc07d7710875b7b")
Coref_model = T5ForConditionalGeneration.from_pretrained("/home/yden079/.cache/huggingface/hub/models--coref--t5/snapshots/f0f21fc4cae5dc130d97e4fa4dc07d7710875b7b")
decomposer_model.eval()
Coref_model.eval()

# Load embedding database and retriever for retrieval
def load_vectorstore(
    model_path: str = "/home/yden079/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    db_path: str = "../data/hotpot_vector_db1",
    k: int = 20
):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"local_files_only": True}
    )

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=db_path
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever

# MMR reranker
def mmr_rerank(query, docs, top_k, lambda_param=1.0):
    
    doc_texts = [doc.page_content for doc in docs]
    pairs = [[query, doc] for doc in doc_texts]
    relevance_scores = rerank_model.compute_score(pairs)

    doc_embeddings = similarity_model.encode(doc_texts, convert_to_tensor=True)
    similarity_matrix = cos_sim(doc_embeddings, doc_embeddings)

    selected_indices = []
    candidate_indices = list(range(len(docs)))

    for _ in range(min(top_k, len(docs))):
        mmr_scores = []

        for idx in candidate_indices:
            relevance = relevance_scores[idx]
            if selected_indices:
                diversity = max([similarity_matrix[idx][j].item() for j in selected_indices])
            else:
                diversity = 0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr_score, idx))

        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        selected_idx = mmr_scores[0][1]

        selected_indices.append(selected_idx)
        candidate_indices.remove(selected_idx)
        
    final_docs = [docs[i] for i in selected_indices]

    # === 返回重排后的 Document 对象 ===
    return final_docs

def deduplicate_docs(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs

def get_prompt_template(dataset_name):
    hotpot_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question. 
You are answering a multi-hop question based on the context provided below.
This means the answer may require combining information from multiple parts of the context.
If the context does not contain enough information, respond with "I don't know."
Your answer should be accurate and briefly.
Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

Context:
{context}

Question:
{question}

Answer:
"""

    pubmed_prompt_template = """
Answer the question based on the context provided below. 
You must answer the question by starting with "yes", "no", or "maybe" and based on the context.
If the context does not contain enough information, respond with "Maybe."
Then provide a brief, accurate explanation in complete sentences.

Context:
{context}

Question:
{question}

Answer:
"""

    msmarco_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question. 
You are answering a question based on the context provided below.
If the context does not contain enough information, respond with "I don't know."
Your answer should be accurate, briefly, and use complete sentences.

Context:
{context}

Question:
{question}

Answer:
"""

    templates = {
        "hotpot": hotpot_prompt_template,
        "pubmed": pubmed_prompt_template,
        "msmarco": msmarco_prompt_template
    }

    if dataset_name not in templates:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Choose from {list(templates.keys())}.")

    return templates[dataset_name]

def safe_llm_invoke(prompt_input, timeout=30):
    result_holder = {}

    def call_llm():
        try:
            result_holder["result"] = llm.invoke(prompt_input)
        except Exception as e:
            result_holder["result"] = None

    thread = threading.Thread(target=call_llm)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None

    if result_holder.get("result") is None:
        return None

    return result_holder["result"].content

def QuestionClassifier(question):
    question = f"classify query complexity: {question}"
    
    input_ids = classifier_tokenizer(
        question,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        outputs = classifier_model.generate(
            input_ids,
            max_length=2,
            num_beams=4,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(
        torch.stack([
            scores[:, classifier_tokenizer('A').input_ids[0]],
            scores[:, classifier_tokenizer('B').input_ids[0]],
            scores[:, classifier_tokenizer('C').input_ids[0]],
        ]), dim=0,
    ).detach().cpu().numpy()
    
    pred_label = np.argmax(probs, 0)[0]
    label_to_option = {0: 'A', 1: 'B', 2: 'C'}
    classification = label_to_option[pred_label]

    return classification

def choose_lambda(qtype):
    q_type = qtype.lower()

    if q_type == "A":
        return 0.9
    elif q_type == "B":
        return 0.6
    else:
        return 0.1

def QuestionDecompositioner(question):

    input_text = f"decompose question: {question}"
    input_ids = decomposer_tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        outputs = decomposer_model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    decoded_output = decomposer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    sub_questions = decoded_output.split(" [SEP] ")

    return sub_questions

def CorefResolver(previous_question, previous_answer, follow_question, llm):
    input_text = "|||".join([previous_question, previous_answer, follow_question])
    input_ids = Coref_tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        outputs = Coref_model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    resolved_question = Coref_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resolved_question

def MultiHopAnswerer(retriever, question: str, llm) -> str:
    """
    question: str,
    answers: List[str]
    """
    sub_questions = QuestionDecompositioner(question)
    all_contexts = []
    resolved_questions = []
    answers = []
    for i, sub_q in enumerate(sub_questions):
        if i == 0:
            resolved_questions.append(sub_q)
            resolved_q = sub_q
        else:
            resolved_q = CorefResolver(resolved_questions[i - 1], answers[i - 1], sub_q, llm)
            resolved_questions.append(resolved_q)
        docs = retriever.get_relevant_documents(resolved_q)
        docs = deduplicate_docs(docs)
        top_docs = mmr_rerank(resolved_q, docs, 5, lambda_param=1)
        context = "\n\n".join([doc.page_content for doc in top_docs])
        all_contexts.append(context)
        prompt = f"""You are an expert of world knowledge. 
I will ask you a question, and you must answer it based only on the context provided below. 
Your answer should be accurate and brief.

Question:
{resolved_q}

Context:
{context}

Answer:
"""
        answer = safe_llm_invoke(prompt, timeout=30)
        answers.append(answer)
    summary_prompt = "Here are the sub-questions and their answers:\n"
    for i, (q, a) in enumerate(zip(resolved_questions, answers), 1):
        summary_prompt += f"sub-question {i}: {q}\n answer {i}: {a}\n"

    all_paragraphs = []
    for ctx in all_contexts:
        for para in ctx.split("\n\n"):
            if para.strip():
                all_paragraphs.append(para.strip())

    seen = set()
    unique_paragraphs = []
    for para in all_paragraphs:
        if para not in seen:
            seen.add(para)
            unique_paragraphs.append(para)

    final_context = "\n\n".join(unique_paragraphs)
    summary_prompt += f"\nContext:\n{final_context}\n"

    summary_prompt += f"\nNow based ONLY on the sub-questions, their answers, and the context above, answer the original question briefly.\n original question: \"{question}\""
    final_answer = safe_llm_invoke(summary_prompt, timeout=30)
    return final_answer, final_context

def normalize(text):
    if text is None:
        return ""
    return text.lower().replace("’", "'").strip()

def run_rag_inference_pipeline(question_file: str, retriever, max_questions: int = 50):
    with open(question_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    total_time =0.0

    lambda_set = []
    results = []

    for i, item in enumerate(tqdm(qa_pairs[:max_questions])):
        q = item["question"]
        gold_answer = item["answer"]
        qtype = QuestionClassifier(q)
        lamda = choose_lambda(qtype)
        start = time.time()
        if qtype == "A" or qtype == "B":
            docs = retriever.get_relevant_documents(q)
            docs = deduplicate_docs(docs)
            top_docs = mmr_rerank(q, docs, top_k=2, lambda_param=lamda)

            top_context = "\n\n".join([doc.page_content for doc in top_docs])
            prompt_template = get_prompt_template("hotpot")
            prompt_input = prompt_template.format(context=top_context, question=q)
            result = safe_llm_invoke(prompt_input, timeout=30)
            if "i don't know." in normalize(result):
                result, top_context = MultiHopAnswerer(retriever, q, llm)
        else:
            result, top_context = MultiHopAnswerer(retriever, q, llm)
        end = time.time()
        total_time += (end - start)

        results.append({
            "question": q,
            "retrieved_context": top_context,
            "generated_answer": result,
            "gold_answer": gold_answer
        })

    print(f"Total Time: {total_time:.3f} seconds")

    return results

vectorstore, retriever = load_vectorstore(model_path="/home/yden079/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",db_path="../data/hotpot_vector_db1",k=20)
results = run_rag_inference_pipeline(question_file="../data/hotpot_qa.json",retriever=retriever)
with open("rag_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)