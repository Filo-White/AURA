"""
Image Analysis Module — wraps the LLM4VISION / AURA pipeline.

Four-agent approach (matching the AURA architecture):
  1. Multimodal Retrieval Agent: find the most similar IPC-A-610F reference
     images for the uploaded PCB photo (via Qdrant multimodal vector store).
  2. Defect Analysis Agent: decide whether the image contains a defect.
     If no defect is detected the workflow ends.
  3. Segmentation Detection Agent: run Sa2VA-4B to produce defect
     segmentation masks (only if a defect was detected).
  4. Explanation Agent: provide a textual explanation of the defect with
     explicit references to the IPC-A-610F book/figures retrieved by
     the Retrieval Agent.

This module is self-contained and does NOT import anything from the
chatbot module.
"""

import os
import atexit
import base64
import json
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import requests
from PIL import Image
import numpy as np

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
DOCS_DIR = _MODULE_DIR / "docs"
QDRANT_DATA_DIR = _MODULE_DIR / "qdrant_data"
STORAGE_DIR = _MODULE_DIR / "storage_dir"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: Dict[str, Dict] = {
    "GPT-5.2 (OpenAI)": {"provider": "openai", "model_id": "gpt-5.2"},
    "Qwen3-VL (Ollama)": {"provider": "ollama", "model_id": "qwen3-vl:latest"},
}

DEFAULT_MODEL = "GPT-5.2 (OpenAI)"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons so Streamlit doesn't reload on every interaction
# ---------------------------------------------------------------------------
_retriever_engine = None
_qdrant_client = None
_openai_llms: Dict[str, object] = {}
_current_manual_id: Optional[str] = None
_manual_name: str = "IPC-A-610F"


def configure(content_dir: str, qdrant_dir: str, storage_dir: str,
              manual_id: str = "", manual_name: str = ""):
    """Point this module at a different manual's data directories.

    Resets the Qdrant retriever singleton so the next call reloads from the
    new paths.  Safe to call multiple times with the same *manual_id* — the
    singleton is only reset when the id actually changes.
    """
    global DOCS_DIR, QDRANT_DATA_DIR, STORAGE_DIR
    global _retriever_engine, _qdrant_client, _current_manual_id, _manual_name

    if manual_id and manual_id == _current_manual_id:
        return  # nothing to do

    # Close old client cleanly
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None
    _retriever_engine = None

    DOCS_DIR = Path(content_dir)
    QDRANT_DATA_DIR = Path(qdrant_dir)
    STORAGE_DIR = Path(storage_dir)
    _current_manual_id = manual_id
    _manual_name = manual_name or manual_id or "reference manual"
    logger.info("image_analysis_module reconfigured for manual '%s'.", manual_id)


def _cleanup_qdrant():
    """Close the Qdrant client cleanly before Python shuts down."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None


atexit.register(_cleanup_qdrant)


def _get_index():
    """Build or load the Qdrant-backed multimodal vector index (singleton)."""
    global _retriever_engine, _qdrant_client

    if _retriever_engine is not None:
        return _retriever_engine

    import qdrant_client
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    logger.info("Loading Qdrant multimodal index …")

    _qdrant_client = qdrant_client.QdrantClient(path=str(QDRANT_DATA_DIR))

    text_store = QdrantVectorStore(client=_qdrant_client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=_qdrant_client, collection_name="image_collection")

    storage_ctx = StorageContext.from_defaults(
        persist_dir=str(STORAGE_DIR),
        vector_store=text_store,
        image_store=image_store,
    )
    index = load_index_from_storage(storage_ctx)
    _retriever_engine = index.as_retriever(image_similarity_top_k=4)
    logger.info("Qdrant multimodal index ready.")
    return _retriever_engine


def _get_openai_llm(model_id: str = "gpt-4o"):
    """Return a cached OpenAI multimodal LLM handle for the given model."""
    global _openai_llms
    if model_id not in _openai_llms:
        from llama_index.llms.openai import OpenAI

        _openai_llms[model_id] = OpenAI(model=model_id, max_new_tokens=1500)
    return _openai_llms[model_id]


def _image_to_base64(path: str) -> str:
    """Read an image file and return its base64 encoding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Defect Analysis Agent  (stage 2)
# ---------------------------------------------------------------------------

def _defect_analysis_prompt() -> str:
    """Build the Defect Analysis Agent prompt using the active manual name."""
    m = _manual_name
    return (
        f"You are an expert {m} inspector. "
        "The FIRST image is a photo to inspect. "
        f"The REMAINING images are {m} reference examples.\n\n"
        "Your response must focus exclusively on the defects (or absence "
        "of defects) visible in the image. "
        "Do NOT comment on the quality or relevance of the reference images. "
        "If the references are not useful, simply ignore them and base your "
        "analysis on the image alone.\n\n"
        "Start your reply with exactly one of the following markers on its "
        "own line:\n"
        "  DEFECT: YES\n"
        "  DEFECT: NO\n\n"
        "Then:\n"
        "1. Briefly describe any defects visible in the image.\n"
        "2. Provide a severity classification (Class 1 / 2 / 3) for each defect.\n"
        "3. If no defects are visible, state that the board appears acceptable."
    )


def _analyze_with_ollama(
    image_path: str,
    reference_paths: List[str],
    model_id: str = "qwen2.5vl",
) -> str:
    """Send images to a local Ollama vision model and return the analysis."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    images = [_image_to_base64(image_path)]
    # Limit references for local models to avoid excessive processing time
    for ref in reference_paths[:2]:
        if os.path.isfile(ref):
            images.append(_image_to_base64(ref))

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": _defect_analysis_prompt(),
                "images": images,
            }
        ],
        "stream": False,
    }

    logger.info(f"Sending {len(images)} image(s) to Ollama model '{model_id}'...")
    resp = requests.post(
        f"{ollama_host}/api/chat",
        json=payload,
        timeout=900,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "No response from model.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_similar_images(
    image_path: str, top_k: int = 4
) -> List[str]:
    """
    Given a PCB image path, retrieve the most similar IPC-A-610F reference
    images from the vector store.

    Returns a deduplicated list of file paths (up to *top_k*).
    """
    retriever = _get_index()
    results = retriever.image_to_image_retrieve(image_path)

    seen = set()
    paths: List[str] = []
    for res in results:
        fp = res.node.metadata.get("file_path", "")
        # Adjust paths that were indexed on Linux
        if fp and not os.path.isfile(fp):
            # Try resolving inside the local docs folder
            basename = os.path.basename(fp)
            local_candidate = str(DOCS_DIR / basename)
            if os.path.isfile(local_candidate):
                fp = local_candidate
        if fp and fp not in seen:
            seen.add(fp)
            paths.append(fp)
        if len(paths) >= top_k:
            break

    return paths


def _parse_has_defect(text: str) -> bool:
    """Return True when the Defect Analysis Agent response indicates a defect."""
    upper = text.upper()
    for line in upper.splitlines():
        stripped = line.strip()
        if stripped.startswith("DEFECT:"):
            return "YES" in stripped
    # Fallback heuristic
    if "DEFECT: NO" in upper or "NO DEFECT" in upper:
        return False
    return True  # default: assume defect when uncertain


def _get_reference_labels(reference_paths: List[str]) -> List[str]:
    """Extract human-readable figure labels from reference file paths.

    E.g. 'docs/Figure_8-119.jpeg' → 'Figure 8-119'.
    """
    labels: List[str] = []
    for p in reference_paths:
        stem = Path(p).stem  # e.g. 'Figure_8-119'
        label = stem.replace("_", " ")
        labels.append(label)
    return labels


def analyze_image(
    image_path: str,
    reference_paths: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[str, bool, List[str]]:
    """
    **Defect Analysis Agent** — decide whether the PCB image contains a
    defect by comparing it against IPC-A-610F reference images.

    Parameters
    ----------
    image_path : str
        Path to the uploaded PCB image.
    reference_paths : list[str] | None
        If *None*, references are retrieved automatically.
    model_name : str
        Key from AVAILABLE_MODELS.

    Returns
    -------
    (analysis_text, has_defect, reference_paths)
    """
    if reference_paths is None:
        reference_paths = retrieve_similar_images(image_path, top_k=4)

    model_info = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS[DEFAULT_MODEL])
    provider = model_info["provider"]
    model_id = model_info["model_id"]

    # --- Ollama (local) path ------------------------------------------------
    if provider == "ollama":
        analysis = _analyze_with_ollama(
            image_path, reference_paths, model_id
        )
        return analysis, _parse_has_defect(analysis), reference_paths

    # --- OpenAI path (matches original RunningExample.ipynb cell 13) --------
    from llama_index.core.llms import (
        ChatMessage,
        ImageBlock,
        TextBlock,
        MessageRole,
    )
    from llama_index.core.schema import ImageDocument

    # Build ImageDocument list (as in original notebook)
    image_documents = [ImageDocument(image_path=image_path)]
    for ref in reference_paths:
        image_documents.append(ImageDocument(image_path=ref))

    llm = _get_openai_llm(model_id)

    # Build multimodal chat message
    blocks = [
        TextBlock(text=_defect_analysis_prompt()),
        ImageBlock(path=image_path),
    ]

    for ref in reference_paths:
        blk = ImageBlock(path=ref)
        if ref.lower().endswith((".jpg", ".jpeg")):
            blk.image_mimetype = "image/jpeg"
        elif ref.lower().endswith(".png"):
            blk.image_mimetype = "image/png"
        blocks.append(blk)

    msg = ChatMessage(role=MessageRole.USER, blocks=blocks)
    response = llm.chat(messages=[msg])
    analysis = str(response)

    return analysis, _parse_has_defect(analysis), reference_paths


# ---------------------------------------------------------------------------
# Explanation Agent  (stage 4)
# ---------------------------------------------------------------------------

def _explain_with_ollama(
    defect_analysis: str,
    segmentation_text: str,
    reference_labels: List[str],
    model_id: str,
) -> str:
    """Call Ollama (text-only) to generate the explanation with references."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    prompt = _build_explanation_prompt(
        defect_analysis, segmentation_text, reference_labels
    )

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    logger.info("Sending explanation request to Ollama …")
    resp = requests.post(
        f"{ollama_host}/api/chat",
        json=payload,
        timeout=900,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "No response from model.")


def _build_explanation_prompt(
    defect_analysis: str,
    segmentation_text: str,
    reference_labels: List[str],
) -> str:
    """Build the prompt for the Explanation Agent."""
    refs_str = ", ".join(reference_labels) if reference_labels else "none"

    seg_block = ""
    if segmentation_text:
        seg_block = (
            f"\n\nSegmentation model findings:\n{segmentation_text}"
        )

    m = _manual_name
    return (
        "You are the Explanation Agent of the AURA inspection system.\n\n"
        "Below you will find:\n"
        "  - The Defect Analysis Agent's assessment of the image.\n"
        "  - (Optionally) the Segmentation Agent's findings.\n"
        f"  - The list of {m} reference figures retrieved from the "
        "knowledge base.\n\n"
        f"--- Defect Analysis ---\n{defect_analysis}\n"
        f"{seg_block}\n"
        f"--- Retrieved {m} References ---\n{refs_str}\n\n"
        "Your task:\n"
        "1. Provide a clear, concise explanation of WHY the image is "
        "classified as defective (or acceptable).\n"
        f"2. Explicitly cite the relevant {m} figures from the "
        "retrieved references listed above (e.g. 'According to Figure "
        "X-Y ...'). Only cite figures that are relevant to the defect "
        "found.\n"
        "3. If the retrieved references are not relevant, do NOT mention "
        "them. Base your explanation solely on the defect analysis.\n"
        "4. End with a short summary and the overall severity class "
        "(Class 1 / 2 / 3).\n"
    )


def explain_defect(
    defect_analysis: str,
    segmentation_text: str = "",
    reference_paths: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    **Explanation Agent** — produce a detailed textual explanation of the
    detected defect, citing specific IPC-A-610F figures from the retrieved
    references.

    Parameters
    ----------
    defect_analysis : str
        Output from the Defect Analysis Agent.
    segmentation_text : str
        Output from the Segmentation Agent (Sa2VA-4B prediction text).
    reference_paths : list[str] | None
        Paths to the retrieved IPC-A-610F reference images.
    model_name : str
        Key from AVAILABLE_MODELS.

    Returns
    -------
    explanation_text : str
    """
    reference_labels = _get_reference_labels(reference_paths or [])

    model_info = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS[DEFAULT_MODEL])
    provider = model_info["provider"]
    model_id = model_info["model_id"]

    if provider == "ollama":
        return _explain_with_ollama(
            defect_analysis, segmentation_text, reference_labels, model_id
        )

    # --- OpenAI path --------------------------------------------------------
    from llama_index.core.llms import (
        ChatMessage,
        TextBlock,
        MessageRole,
    )

    llm = _get_openai_llm(model_id)
    prompt = _build_explanation_prompt(
        defect_analysis, segmentation_text, reference_labels
    )
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text=prompt)],
    )
    response = llm.chat(messages=[msg])
    return str(response)


# ---------------------------------------------------------------------------
# Segmentation — Sa2VA-4B
# ---------------------------------------------------------------------------
_sa2va_model = None
_sa2va_tokenizer = None


def _get_sa2va():
    """Load the Sa2VA-4B segmentation model (singleton).

    Load in bfloat16 on CUDA when available,
    otherwise float32 on CPU.
    """
    global _sa2va_model, _sa2va_tokenizer

    if _sa2va_model is not None:
        return _sa2va_model, _sa2va_tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModel

    model_path = "ByteDance/Sa2VA-4B"
    logger.info(f"Loading Sa2VA-4B from {model_path} …")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    _sa2va_model = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,              # (al posto di torch_dtype=...)
        trust_remote_code=True,
        use_flash_attn=False,     # non richiedere il pacchetto flash_attn esterno
    )

    if use_cuda:
        _sa2va_model = _sa2va_model.cuda()

    _sa2va_model = _sa2va_model.eval()
    
    _sa2va_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    logger.info(
        "Sa2VA-4B loaded successfully (device=%s).",
        "cuda" if use_cuda else "cpu",
    )
    return _sa2va_model, _sa2va_tokenizer

def segment_image(
    image_path: str,
    prompt: str = (
        "<image>Find soldering errors due to excess solder."
        "Please respond with interleaved segmentation masks "
        "for the corresponding parts of the answer."
    ),
) -> Tuple[str, Optional[Image.Image]]:
    """
    Run Sa2VA-4B segmentation on the given image.

    Parameters
    ----------
    image_path : str
        Path to the PCB image.
    prompt : str
        The text prompt for the segmentation model.

    Returns
    -------
    (prediction_text, overlay_image)
        *overlay_image* is a PIL Image with coloured masks drawn on top
        of the original, or *None* if no masks were produced.
    """
    model, tokenizer = _get_sa2va()

    image = Image.open(image_path).convert("RGB")
    input_dict = {
        "image": image,
        "text": prompt,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }

    return_dict = model.predict_forward(**input_dict)
    answer = return_dict["prediction"]
    masks = return_dict.get("prediction_masks")  # list[np.array(1, h, w)]

    if not masks:
        return answer, None

    # Build a coloured overlay --------------------------------------------------
    overlay = np.array(image, dtype=np.float64)
    # Distinct colours for up to 10 masks
    palette = [
        (255, 50, 50),    # red
        (50, 180, 255),   # blue
        (50, 255, 100),   # green
        (255, 200, 50),   # yellow
        (200, 50, 255),   # purple
        (255, 130, 50),   # orange
        (50, 255, 220),   # cyan
        (255, 50, 200),   # pink
        (130, 255, 50),   # lime
        (50, 100, 255),   # indigo
    ]
    alpha = 0.45

    for idx, mask in enumerate(masks):
        m = mask.squeeze()  # (h, w)
        if m.shape != overlay.shape[:2]:
            from PIL import Image as _PILResize

            m_img = _PILResize.fromarray((m * 255).astype(np.uint8))
            m_img = m_img.resize(
                (overlay.shape[1], overlay.shape[0]),
                resample=_PILResize.NEAREST,
            )
            m = np.array(m_img) / 255.0

        colour = palette[idx % len(palette)]
        for c in range(3):
            overlay[:, :, c] = np.where(
                m > 0.5,
                overlay[:, :, c] * (1 - alpha) + colour[c] * alpha,
                overlay[:, :, c],
            )

    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    return answer, overlay_img