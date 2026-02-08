"""QA runner that performs RAG using the FAISS index and OpenAI completions.

Functions:
- retrieve(query, index, metadatas, embeddings, model)
- retrieve_and_generate(query, docs, openai_api_key=None)

This uses a simple prompt template and OpenAI's text completion API if API key is provided. Otherwise it will return the concatenated context and the query.
"""
import os
import faiss
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


def retrieve_semantic_search(query: str, model, index: faiss.IndexFlatIP, metadatas: List[Dict], k_top: int = 5):
    """
    Retrieve the top-k most relevant document chunks for a text query using a FAISS index.

    Parameters
    - query (str): The query text to embed and search for.
    - model: Embedding model with an `encode(..., convert_to_numpy=True, normalize_embeddings=True)` method
             (e.g., a SentenceTransformer) that returns a numpy embedding for the query.
    - index (faiss.IndexFlatIP): FAISS index storing normalized embeddings using inner-product similarity.
    - metadatas (List[Dict]): List of metadata dictionaries corresponding to each indexed vector. The list
                              must be aligned with the vectors stored in the index (same order and length).
    - k_top (int): Number of top results to return (default: 5).

    Returns
    - results: A list of result dictionaries sorted by relevance. Each dictionary contains:
        - "score" (float): similarity score (inner-product) between query and chunk embedding.
        - "metadata" (Dict): the metadata associated with the returned chunk.
        - "idx" (int): the integer index of the chunk within the original corpus/index.
    """
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k_top)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(score), "metadata": metadatas[idx], "idx": int(idx)})
    return results

def generate_llm_response(query: str, index, metadatas, retrieved_chunks: List[dict], texts: List[str], openai_api_key: str = None) -> str:
    """
    Run a RAG-style question-answer step using a FAISS index and optional OpenAI completion.

    Behavior
    - If openai_api_key is provided, calls OpenAI Completion API with a legal-assistant prompt to produce a concise
      answer citing sources; otherwise returns the assembled context and the query for offline/local handling.

    Parameters
    - query (str): The user question.
    - index: FAISS index containing normalized embeddings.
    - metadatas (List[Dict]): List of metadata dicts aligned with indexed vectors.
    - retrieved_chunks (List[Dict]): list of dictioaries, each representing a chunk of text selected by the retriever module
    - texts (List[str]): The original text chunks indexed; used to build the context returned to the LLM or user.
    - openai_api_key (str, optional): OpenAI API key. If provided, the function will call OpenAI to generate a final answer.

    Returns
    - str: If OpenAI key is provided, the LLM-generated answer (string). Otherwise a string containing the
           assembled context and the original query for downstream processing.
    """
    context_parts = []
    for r in retrieved_chunks:
        context_parts.append(texts[r['idx']])
    context = "\n---\n".join(context_parts)
    if openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                # This is the default and can be omitted
                api_key=openai_api_key,
            )

            openai_instructions = (
                "You are a legal assistant. Use the context to answer the question.\n"
                "Context:\n"
                f"{context}\n\n"
                "Question: "
                f"{query}\n\n"
                "Answer concisely and cite sources by filename and chunk index."
            )
            resp = client.responses.create(
                model='gpt-5.1',
                input=context,
                instructions=openai_instructions
            )
            return resp.output[0].content[0].text
        except Exception as e:
            return f"OpenAI call failed: {e}\n\nContext:\n{context}"
    else:
        return f"--- CONTEXT ---\n{context}\n\n--- QUERY ---\n{query}"

def retrieve_and_generate(query: str, model, index, metadatas, texts: List[str], openai_api_key: str = None, k_top: int = 5) -> str:
    """
    Run a RAG-style retrieval step followed by question-answer step using a FAISS index and optional OpenAI completion.

    Behavior
    - Performs semantic retrieval (top-k) for the query using the provided embedding model and FAISS index.
    - Concatenates retrieved text chunks into a context.
    - If openai_api_key is provided, calls OpenAI Completion API with a legal-assistant prompt to produce a concise
      answer citing sources; otherwise returns the assembled context and the query for offline/local handling.

    Parameters
    - query (str): The user question.
    - model: Embedding model used to encode the query (must support ``encode(..., convert_to_numpy=True, normalize_embeddings=True)``).
    - index: FAISS index containing normalized embeddings.
    - metadatas (List[Dict]): List of metadata dicts aligned with indexed vectors.
    - texts (List[str]): The original text chunks indexed; used to build the context returned to the LLM or user.
    - openai_api_key (str, optional): OpenAI API key. If provided, the function will call OpenAI to generate a final answer.
    - k_top (int): Number of top results to retrieve for context (default: 5).

    Returns
    - llm_response: output from generate_llm_response()
    """
    retrieved_chunks = retrieve_semantic_search(query, model, index, metadatas, k_top)
    llm_response = generate_llm_response(query, index, metadatas, retrieved_chunks, texts, openai_api_key)

    return llm_response
