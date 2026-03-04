"""
Manual Manager — handles PDF ingestion, image extraction, and index building.

Makes the system manual-agnostic: any PDF manual can be uploaded, parsed,
and vectorised.  If it was already processed the existing indices are reused.

Per-manual directory layout (inside ``manuals/<manual_id>/``):
    content/        PDF + extracted figures (input for SimpleDirectoryReader)
    qdrant_data/    Qdrant multimodal vector store
    storage_dir/    LlamaIndex persisted storage context
    vector_db/      FAISS text index for chatbot
    cache/images/   page-level image cache for chatbot references
    manifest.json   marks processing as complete
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
MANUALS_DIR = _MODULE_DIR / "manuals"

# Legacy paths (pre-existing IPC-A-610F data)
_LEGACY_CONTENT_DIR = _MODULE_DIR / "docs"
_LEGACY_QDRANT_DIR = _MODULE_DIR / "qdrant_data"
_LEGACY_STORAGE_DIR = _MODULE_DIR / "storage_dir"
_LEGACY_VECTOR_DB_DIR = _MODULE_DIR / "vector_db"
_LEGACY_CACHE_DIR = _MODULE_DIR / "cache"
_LEGACY_PDF_NAME = "IPC-A-610F.pdf"


# ── helpers ────────────────────────────────────────────────────────────────

def _sanitize_id(name: str) -> str:
    """Turn a filename into a safe directory name."""
    stem = Path(name).stem
    clean = re.sub(r"[^\w\s-]", "", stem).strip()
    clean = re.sub(r"\s+", "_", clean)
    return clean or "manual"


# ── PDF relevance validation ───────────────────────────────────────────────

# Domain keywords grouped by category.  A PDF must hit at least
# MIN_KEYWORD_HITS distinct keywords across at least MIN_CATEGORIES
# categories to be accepted.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "pcb_general": [
        "pcb", "printed circuit", "circuit board", "printed board",
        "printed wiring", "pwb",
    ],
    "soldering": [
        "solder", "soldering", "reflow", "wave solder", "flux",
        "wetting", "dewetting", "tinning", "intermetallic",
    ],
    "components": [
        "smd", "smt", "through-hole", "bga", "qfp", "chip component",
        "resistor", "capacitor", "inductor", "connector", "lead",
        "termination", "land", "pad", "via", "barrel",
    ],
    "inspection": [
        "defect", "inspection", "acceptability", "workmanship",
        "anomaly", "nonconforming", "reject", "class 1", "class 2",
        "class 3", "target condition",
    ],
    "standards": [
        "ipc", "a-610", "j-std", "iec", "mil-std", "jedec",
        "ipc-a-610", "ipc-a-620", "ipc-7711", "ipc-7721",
    ],
    "assembly": [
        "assembly", "rework", "repair", "laminate", "substrate",
        "conformal coating", "stencil", "paste", "placement",
        "pick and place",
    ],
}

_MIN_KEYWORD_HITS = 6    # distinct keywords found
_MIN_CATEGORIES = 2      # must span at least 2 categories
_PAGES_TO_SCAN = 20      # first N pages to sample


def validate_pdf_relevance(pdf_source) -> tuple:
    """Check whether *pdf_source* is a PCB / electronics-related document.

    Parameters
    ----------
    pdf_source : str | bytes
        File path or raw PDF bytes.

    Returns
    -------
    (is_valid, reason) : (bool, str)
    """
    import fitz

    try:
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))
    except Exception as e:
        return False, f"Cannot open PDF: {e}"

    if len(doc) == 0:
        doc.close()
        return False, "The PDF has no pages."

    # Extract text from first N pages
    pages_to_read = min(len(doc), _PAGES_TO_SCAN)
    full_text = ""
    for i in range(pages_to_read):
        full_text += doc[i].get_text() + "\n"
    doc.close()

    text_lower = full_text.lower()

    hits: set = set()
    categories_hit: set = set()

    for category, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                hits.add(kw)
                categories_hit.add(category)

    if len(hits) >= _MIN_KEYWORD_HITS and len(categories_hit) >= _MIN_CATEGORIES:
        return True, (
            f"Accepted — found {len(hits)} domain keywords "
            f"across {len(categories_hit)} categories."
        )

    return False, (
        f"This PDF does not appear to be a PCB / electronics manufacturing "
        f"manual.  Found only {len(hits)} relevant keyword(s) in "
        f"{len(categories_hit)} category(ies) "
        f"(need >= {_MIN_KEYWORD_HITS} keywords in >= {_MIN_CATEGORIES} "
        f"categories).  Matched: {', '.join(sorted(hits)) or 'none'}."
    )


def _read_manifest(path: Path) -> Optional[Dict]:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_manifest(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── public queries ─────────────────────────────────────────────────────────

def list_processed_manuals() -> List[str]:
    """Return manual IDs that have been fully processed."""
    _ensure_legacy_registered()
    ids: List[str] = []
    if MANUALS_DIR.is_dir():
        for child in sorted(MANUALS_DIR.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                ids.append(child.name)
    return ids


def is_manual_processed(manual_id: str) -> bool:
    manifest = MANUALS_DIR / manual_id / "manifest.json"
    return manifest.is_file()


def get_manual_paths(manual_id: str) -> Dict[str, str]:
    """
    Return a dict with absolute string paths for the given manual:
        content_dir, qdrant_dir, storage_dir, vector_db_dir, cache_dir, pdf_path
    """
    manifest_path = MANUALS_DIR / manual_id / "manifest.json"
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Manual '{manual_id}' not found or not processed.")

    # Paths stored in manifest are relative to _MODULE_DIR
    base = _MODULE_DIR
    return {
        "content_dir":   str(base / manifest["content_dir"]),
        "qdrant_dir":    str(base / manifest["qdrant_dir"]),
        "storage_dir":   str(base / manifest["storage_dir"]),
        "vector_db_dir": str(base / manifest["vector_db_dir"]),
        "cache_dir":     str(base / manifest["cache_dir"]),
        "pdf_path":      str(base / manifest["pdf_path"]),
        "manual_name":   manifest.get("manual_name", manual_id),
    }


# ── legacy auto-detection ──────────────────────────────────────────────────

def _ensure_legacy_registered():
    """If old IPC-A-610F dirs exist but no manifest, register them."""
    legacy_pdf = _LEGACY_CONTENT_DIR / _LEGACY_PDF_NAME
    legacy_id = _sanitize_id(_LEGACY_PDF_NAME)
    manifest_path = MANUALS_DIR / legacy_id / "manifest.json"

    if manifest_path.is_file():
        return  # already registered

    # Check all four pieces exist
    if not (legacy_pdf.is_file()
            and _LEGACY_QDRANT_DIR.is_dir()
            and _LEGACY_STORAGE_DIR.is_dir()
            and _LEGACY_VECTOR_DB_DIR.is_dir()):
        return  # not enough data

    logger.info("Registering legacy IPC-A-610F manual …")
    manifest = {
        "manual_id":   legacy_id,
        "manual_name": "IPC-A-610F",
        "original_filename": _LEGACY_PDF_NAME,
        "content_dir":   _LEGACY_CONTENT_DIR.relative_to(_MODULE_DIR).as_posix(),
        "qdrant_dir":    _LEGACY_QDRANT_DIR.relative_to(_MODULE_DIR).as_posix(),
        "storage_dir":   _LEGACY_STORAGE_DIR.relative_to(_MODULE_DIR).as_posix(),
        "vector_db_dir": _LEGACY_VECTOR_DB_DIR.relative_to(_MODULE_DIR).as_posix(),
        "cache_dir":     _LEGACY_CACHE_DIR.relative_to(_MODULE_DIR).as_posix(),
        "pdf_path":      (legacy_pdf).relative_to(_MODULE_DIR).as_posix(),
        "legacy": True,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, manifest)
    logger.info(f"Legacy manual registered as '{legacy_id}'.")


# ── PDF image extraction (from RunningExample.ipynb cell 1) ────────────────

def _extract_images_from_pdf(pdf_path: str, output_dir: str) -> int:
    """Extract embedded images from *pdf_path*, save to *output_dir*.

    Returns the number of images saved.
    """
    import fitz

    os.makedirs(output_dir, exist_ok=True)

    def sanitize_filename(text: str, max_len: int = 100) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s-]", "", text)
        text = text.strip().replace(" ", "_")
        return text[:max_len] if text else "untitled"

    def block_text(block) -> str:
        parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
        return " ".join(parts).strip()

    def find_caption_below(img_rect, text_blocks, max_dy=120):
        candidates = []
        for b in text_blocks:
            r = fitz.Rect(b["bbox"])
            if r.y0 < img_rect.y1:
                continue
            dy = r.y0 - img_rect.y1
            if dy > max_dy:
                continue
            overlap = min(img_rect.x1, r.x1) - max(img_rect.x0, r.x0)
            if overlap <= 0:
                continue
            txt = block_text(b)
            if not txt:
                continue
            score = dy
            is_fig = bool(re.match(r"^(Figure|Fig\.?|FIGURE)\s*\d+", txt))
            candidates.append((0 if is_fig else 1, score, txt))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    doc = fitz.open(pdf_path)
    used_names: set = set()
    count = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_dict = page.get_text("dict")
        text_blocks = [b for b in page_dict["blocks"] if b["type"] == 0]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            img_bytes = base["image"]
            img_ext = base["ext"]

            rects = page.get_image_rects(xref)
            if not rects:
                rects = [None]

            for inst_idx, rect in enumerate(rects):
                caption = None
                if rect is not None:
                    caption = find_caption_below(rect, text_blocks)

                if caption:
                    base_name = sanitize_filename(caption)
                else:
                    base_name = f"page{page_idx+1}_img{img_idx+1}_inst{inst_idx+1}"

                name = base_name
                k = 1
                while name in used_names:
                    name = f"{base_name}_{k}"
                    k += 1
                used_names.add(name)

                out_path = os.path.join(output_dir, f"{name}.{img_ext}")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                count += 1

    doc.close()
    logger.info(f"Extracted {count} images from PDF.")
    return count


# ── Qdrant multimodal index building (from notebook cell 4) ───────────────

def _build_qdrant_index(content_dir: str, qdrant_dir: str, storage_dir: str):
    """Build a Qdrant-backed MultiModalVectorStoreIndex from *content_dir*."""
    import qdrant_client
    from llama_index.core import SimpleDirectoryReader, StorageContext
    from llama_index.core.indices import MultiModalVectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    os.makedirs(qdrant_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)

    client = qdrant_client.QdrantClient(path=qdrant_dir)

    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    logger.info(f"Loading documents from {content_dir} …")
    documents = SimpleDirectoryReader(content_dir).load_data()
    logger.info(f"Indexing {len(documents)} documents into Qdrant …")

    MultiModalVectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    storage_context.persist(persist_dir=storage_dir)
    client.close()
    logger.info("Qdrant multimodal index built and persisted.")


# ── FAISS text index building (from chatbot_module) ───────────────────────

def _build_faiss_index(pdf_path: str, vector_db_dir: str):
    """Build a FAISS text index from the PDF for the chatbot RAG."""
    import fitz

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings

    os.makedirs(vector_db_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        documents.append(
            {"content": text, "metadata": {"source": pdf_path, "page": page_num + 1}}
        )
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_splits = []
    for d in documents:
        chunks = splitter.create_documents([d["content"]], [d["metadata"]])
        all_splits.extend(chunks)

    vs = FAISS.from_documents(all_splits, embeddings)
    vs.save_local(vector_db_dir)
    logger.info("FAISS text index built and persisted.")


# ── main processing entry-point ───────────────────────────────────────────

def process_manual(
    pdf_source,
    original_filename: str,
    progress_callback=None,
) -> str:
    """
    Full pipeline: extract images → build Qdrant index → build FAISS index.

    Parameters
    ----------
    pdf_source : str | bytes | BinaryIO
        File path, raw bytes, or file-like object of the PDF.
    original_filename : str
        Original filename (used to derive the manual ID).
    progress_callback : callable | None
        ``callback(step: int, total: int, message: str)`` for UI updates.

    Returns
    -------
    manual_id : str
    """
    manual_id = _sanitize_id(original_filename)
    manual_dir = MANUALS_DIR / manual_id
    manifest_path = manual_dir / "manifest.json"

    if manifest_path.is_file():
        logger.info(f"Manual '{manual_id}' already processed — skipping.")
        return manual_id

    content_dir = manual_dir / "content"
    qdrant_dir = manual_dir / "qdrant_data"
    storage_dir = manual_dir / "storage_dir"
    vector_db_dir = manual_dir / "vector_db"
    cache_dir = manual_dir / "cache"

    content_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save PDF to content dir
    pdf_dest = content_dir / original_filename
    if isinstance(pdf_source, (str, Path)):
        shutil.copy2(str(pdf_source), str(pdf_dest))
    elif isinstance(pdf_source, bytes):
        pdf_dest.write_bytes(pdf_source)
    else:
        pdf_dest.write_bytes(pdf_source.read())

    _notify(progress_callback, 1, 4, "Extracting images from PDF …")
    _extract_images_from_pdf(str(pdf_dest), str(content_dir))

    _notify(progress_callback, 2, 4, "Building multimodal vector index (Qdrant) …")
    _build_qdrant_index(str(content_dir), str(qdrant_dir), str(storage_dir))

    _notify(progress_callback, 3, 4, "Building text index (FAISS) …")
    _build_faiss_index(str(pdf_dest), str(vector_db_dir))

    # 4. Write manifest
    manifest = {
        "manual_id":   manual_id,
        "manual_name": Path(original_filename).stem.replace("_", " ").replace("-", " "),
        "original_filename": original_filename,
        "content_dir":   content_dir.relative_to(_MODULE_DIR).as_posix(),
        "qdrant_dir":    qdrant_dir.relative_to(_MODULE_DIR).as_posix(),
        "storage_dir":   storage_dir.relative_to(_MODULE_DIR).as_posix(),
        "vector_db_dir": vector_db_dir.relative_to(_MODULE_DIR).as_posix(),
        "cache_dir":     cache_dir.relative_to(_MODULE_DIR).as_posix(),
        "pdf_path":      pdf_dest.relative_to(_MODULE_DIR).as_posix(),
        "legacy": False,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, manifest)
    _notify(progress_callback, 4, 4, "Done!")
    logger.info(f"Manual '{manual_id}' processed successfully.")
    return manual_id


def _notify(cb, step, total, msg):
    if cb is not None:
        cb(step, total, msg)
