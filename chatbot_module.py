"""
Chatbot Module — manual-agnostic RAG chatbot.

Provides a RAG-powered chatbot that answers questions about any loaded
reference manual, using:
  - FAISS vector store (pre-built or built on upload) with HuggingFace embeddings
  - A local LLM served via Ollama
  - Text extracted from the manual PDF via PyMuPDF

Call ``configure()`` to point the module at a specific manual’s data.
This module is self-contained and does NOT import anything from the
image analysis module.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — defaults point to legacy IPC-A-610F data; call configure() to
# switch to a different manual.
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
PDF_PATH = _MODULE_DIR / "docs" / "IPC-A-610F.pdf"
VECTOR_DB_PATH = _MODULE_DIR / "vector_db"
CACHE_DIR = _MODULE_DIR / "cache"

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_vector_store = None
_page_content: Dict[int, str] = {}
_reference_map: Dict[str, int] = {}
_page_images: Dict[int, List[str]] = {}
_ollama_model_name: Optional[str] = None
_current_manual_id: Optional[str] = None


def configure(pdf_path: str, vector_db_dir: str, cache_dir: str,
              manual_id: str = ""):
    """Point this module at a different manual's data directories.

    Resets all singletons so the next call rebuilds from the new paths.
    Safe to call repeatedly with the same *manual_id*.
    """
    global PDF_PATH, VECTOR_DB_PATH, CACHE_DIR
    global _vector_store, _page_content, _reference_map, _page_images
    global _current_manual_id

    if manual_id and manual_id == _current_manual_id:
        return

    _vector_store = None
    _page_content = {}
    _reference_map = {}
    _page_images = {}

    PDF_PATH = Path(pdf_path)
    VECTOR_DB_PATH = Path(vector_db_dir)
    CACHE_DIR = Path(cache_dir)
    _current_manual_id = manual_id
    logger.info("chatbot_module reconfigured for manual '%s'.", manual_id)


def _get_vector_store():
    """Load the pre-built FAISS index (singleton)."""
    global _vector_store, _page_content, _reference_map, _page_images

    if _vector_store is not None:
        return _vector_store

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings

    embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    db_path = str(VECTOR_DB_PATH)
    if os.path.isdir(db_path) and os.listdir(db_path):
        logger.info("Loading existing FAISS vector store …")
        _vector_store = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        logger.info("Building FAISS vector store from PDF …")
        _vector_store = _build_vector_store(embeddings, db_path)

    # Also extract page-level metadata for references
    _extract_pdf_metadata()

    logger.info("FAISS vector store ready.")
    return _vector_store


def _build_vector_store(embeddings, db_path: str):
    """Build the FAISS index from the IPC-A-610F PDF."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS

    import fitz

    pdf_path = str(PDF_PATH)
    doc = fitz.open(pdf_path)

    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        documents.append(
            {"content": text, "metadata": {"source": pdf_path, "page": page_num + 1}}
        )
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_splits = []
    for d in documents:
        chunks = splitter.create_documents([d["content"]], [d["metadata"]])
        all_splits.extend(chunks)

    vs = FAISS.from_documents(all_splits, embeddings)
    os.makedirs(db_path, exist_ok=True)
    vs.save_local(db_path)
    return vs


def _extract_pdf_metadata():
    """Extract page content and figure/table references from the PDF."""
    global _page_content, _reference_map, _page_images

    if _page_content:
        return  # already done

    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not available — metadata extraction skipped.")
        return

    pdf_path = str(PDF_PATH)
    if not os.path.isfile(pdf_path):
        logger.warning(f"PDF not found at {pdf_path}")
        return

    image_cache = str(CACHE_DIR / "images")
    os.makedirs(image_cache, exist_ok=True)

    doc = fitz.open(pdf_path)
    figure_pattern = re.compile(r"(Figure|Table)\s+(\d+-\d+)")

    for page_num, page in enumerate(doc):
        text = page.get_text()
        _page_content[page_num + 1] = text

        for m in figure_pattern.finditer(text):
            ref_id = f"{m.group(1)} {m.group(2)}"
            _reference_map[ref_id] = page_num + 1

        img_list = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            fname = f"page_{page_num+1}_img_{img_idx}.{base['ext']}"
            fpath = os.path.join(image_cache, fname)
            if not os.path.isfile(fpath):
                with open(fpath, "wb") as f:
                    f.write(base["image"])
            img_list.append(fpath)
        if img_list:
            _page_images[page_num + 1] = img_list

    doc.close()


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def _query_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Send a prompt to a local Ollama instance and return the response."""
    import requests

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError:
        return (
            "⚠️ Cannot connect to Ollama. "
            "Make sure Ollama is running (`ollama serve`) and the model is pulled."
        )
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"Error communicating with Ollama: {e}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_ollama_model(model_name: str):
    """Set which Ollama model to use (e.g. 'llama3.2', 'mistral', …)."""
    global _ollama_model_name
    _ollama_model_name = model_name


def get_ollama_model() -> str:
    return _ollama_model_name or "llama3.2"


def list_available_ollama_models() -> List[str]:
    """Query Ollama for locally available models."""
    import requests

    try:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{ollama_host}/api/tags", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def search_ipc_standard(query: str, k: int = 5) -> List[Dict]:
    """
    Search the IPC-A-610F vector store for chunks relevant to *query*.
    Returns a list of dicts with keys: content, page, score.
    """
    vs = _get_vector_store()
    results = vs.similarity_search_with_score(query, k=k)
    formatted = []
    for doc, score in results:
        formatted.append(
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "score": float(score),
            }
        )
    return formatted


def _manual_label() -> str:
    """Derive a human-readable label from the current PDF path."""
    return PDF_PATH.stem.replace("_", " ").replace("-", " ")


def answer_question(query: str) -> str:
    """
    Full RAG pipeline: retrieve relevant chunks from the loaded manual,
    build a context-augmented prompt, and generate an answer via Ollama.
    """
    label = _manual_label()
    search_results = search_ipc_standard(query, k=5)

    if not search_results:
        return f"I couldn't find relevant information in the {label} manual for your question."

    context_parts = []
    page_refs = set()
    for r in search_results:
        page_refs.add(r["page"])
        context_parts.append(f"[Page {r['page']}]: {r['content']}")

    # Check for figure / table references
    fig_pattern = re.compile(r"(Figure|Table)\s+(\d+-\d+)")
    referenced_figs = []
    for ctx in context_parts:
        for m in fig_pattern.finditer(ctx):
            ref_id = f"{m.group(1)} {m.group(2)}"
            if ref_id in _reference_map:
                ref_page = _reference_map[ref_id]
                page_refs.add(ref_page)
                referenced_figs.append(f"{ref_id} (page {ref_page})")

    context_str = "\n\n".join(context_parts)

    prompt = f"""You are a technical expert assistant specializing in the {label} standard.
Answer the following question based on the manual information provided below.

Question: {query}

Relevant information from {label} manual:
{context_str}

If there are relevant figures, tables or images mentioned in the text, reference them specifically in your answer.
Referenced items: {", ".join(referenced_figs) if referenced_figs else "None specifically mentioned."}

Provide a detailed answer with page references (in the format [Page X]) whenever possible.
If you're uncertain or the information is not provided in the context, state that clearly.
"""

    model = get_ollama_model()
    answer = _query_ollama(prompt, model=model)

    pages_str = ", ".join(str(p) for p in sorted(page_refs))
    return f"**Based on {label} (pages {pages_str}):**\n\n{answer}"


def get_reference_images_for_pages(pages: List[int]) -> List[Tuple[int, str]]:
    """Return (page, image_path) pairs for the given page numbers."""
    _extract_pdf_metadata()
    out = []
    for p in pages:
        for img in _page_images.get(p, []):
            out.append((p, img))
    return out
