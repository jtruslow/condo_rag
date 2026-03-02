"""Ingest documents (PDF or text), chunk them, compute embeddings, and store a FAISS index with metadata.

This module provides:
- load_documents(paths): returns list of {'text':..., 'source':...}
- chunk_text(text, chunk_size=800, overlap=200)
- build_index(docs, model_name='sentence-transformers/all-MiniLM-L6-v2') -> (faiss_index, metadatas, embeddings)
- save_index(index, metadatas, path)
- load_index(path)

This is intentionally small and synchronous for clarity.
"""
from typing import List, Dict, Tuple, Union
import os
import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import pdfplumber


def read_pdf(p: str) -> str:
    """
    Read text from a PDF file 
    If the ratio of text to images is 9/1 by area, then extract the text; otherwise return "TOO_MANY_IMAGES".

    Parameters
    - p (str): Path to the PDF file.

    Returns
    - str: Extracted text from the PDF
    """

    MAX_ALLOWED_IMAGE_AREA_RATIO = 0.10

    with pdfplumber.open(p) as pdf:
        total_text_area = 0.0
        total_img_area = 0.0
        total_area = 0.0
        
        for page in pdf.pages:
            img_lst = page.images
            char_lst = page.chars
            
            text_area = sum(c['width'] * c['height'] for c in char_lst)
            total_text_area += text_area
            
            img_area = sum(img['width'] * img['height'] for img in img_lst)
            total_img_area += img_area
            
            total_area += page.width * page.height
        
        if total_area > 0 and (total_img_area / (total_img_area + total_text_area)) < MAX_ALLOWED_IMAGE_AREA_RATIO:
            text = []
            for page in pdf.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        else:
            return "TOO_MANY_IMAGES"


def read_txt(p: str) -> str:
    """
    Read text from a plain text file.

    Parameters
    - p (str): Path to the text file.

    Returns
    - str: Contents of the text file.
    """
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def load_documents(paths: Union[List[str], str, os.PathLike]) -> List[Dict]:
    """
    Load documents from the given file paths and return a list of document dictionaries.

    Parameters
    - paths (Union[List[str], str, os.PathLike]): Either a list of filesystem paths to load
        or a single path (string or Path-like) pointing to a single file, 
        the contents of which are a newline-separated list of paths to the desired content
        that you want to ingest.

    Returns
    - docs: A list of dicts, one per successfully read file, each containing:
        - 'text' (str): the extracted text content for the document.
        - 'source' (str): the original file path.

    Behavior
    - For PDF files, uses read_pdf to extract text.
    - For text files, uses read_txt to read contents.
    - On any read error the function prints an error message and skips that file.
    """
    if isinstance(paths, (str, os.PathLike)):
        with open(paths, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip()]
    docs = []
    for p in paths:
        try:
            if p.lower().endswith('.pdf'):
                text = read_pdf(p)
            elif p.lower().endswith('.txt'):
                text = read_txt(p)
            else:
                raise ValueError(f"Unsupported file type for {p}")
            docs.append({"text": text, "source": p})
        except Exception as e:
            print(f"Failed reading {p}: {e}")
    return docs


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 64) -> List[str]:
    """Split text into overlapping token chunks for embedding generation.

    Text is tokenized by splitting on whitespace so punctuation stays attached to adjacent words.

    Parameters
    - text (str): Raw document text to split.
    - chunk_size (int): Max token count per chunk.
    - overlap (int): Token overlap between sequential chunks.

    Returns
    - List[str]: Token chunks preserving context via overlap.
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def build_index(
    docs: List[Dict], 
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 256,
    overlap: int = 64
    ) -> Tuple[faiss.IndexFlatIP, List[Dict], np.ndarray]:
    """Create embeddings from documents and store them in a FAISS index.

    Parameters
    - docs (List[Dict]): Documents with 'text' and 'source' keys to ingest.
    - model_name (str): SentenceTransformer checkpoint for embedding generation.
    - chunk_size (int): Number of tokens per chunk when splitting documents.
    - overlap (int): Number of tokens to overlap between chunks.

    Returns
    - index (faiss.IndexFlatIP): FAISS index populated with document embeddings.
    - metadatas (List[Dict]): Metadata entries parallel to the embeddings.
    - embeddings (np.ndarray): Numeric vectors used to build the index.
    - texts (List[str]): Text chunks that correspond to each embedding.
    """
    model = SentenceTransformer(model_name)
    texts = []
    metadatas = []
    for d in docs:
        chunks = chunk_text(d['text'], chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d.get('source'), "chunk": i})
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, metadatas, embeddings, texts


def save_index(index, metadatas, texts, path: str):
    """Persist the FAISS index, metadata, and texts to disk.

    Parameters
    - index: FAISS index produced by build_index.
    - metadatas (List[Dict]): List of metadata entries, produced by build_index(),
         matching the saved embeddings.
    - texts (List[str]): Text chunks corresponding to each embedding. Produced by build_index()
    - path (str): Directory where index.faiss, metadatas.json, and texts.json are written.
    """
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, 'index.faiss'))
    import json
    with open(os.path.join(path, 'metadatas.json'), 'w', encoding='utf-8') as f:
        json.dump(metadatas, f)
    with open(os.path.join(path, 'texts.json'), 'w', encoding='utf-8') as f:
        json.dump(texts, f)


def load_index(path: str):
    """Load a previously saved FAISS index along with its metadata and texts.

    Parameters
    - path (str): Directory where index.faiss, metadatas.json, and texts.json are stored.

    Returns
    - index: Restored FAISS index instance.
    - metadatas (List[Dict]): Metadata entries previously saved alongside the index.
    - texts (List[str]): Text chunks that correspond to each vector in the index.
    """
    import json
    index = faiss.read_index(os.path.join(path, 'index.faiss'))
    with open(os.path.join(path, 'metadatas.json'), 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    with open(os.path.join(path, 'texts.json'), 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return index, metadatas, texts
