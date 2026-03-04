"""
AURA — Augmented Understanding for Reliable Anomaly-detection
===============================================================
Manual-agnostic inspection assistant with two modules:
  1. Image Analysis — multimodal RAG + LLM defect detection
  2. Manual Chatbot — local Ollama LLM with RAG over the loaded manual
"""

import streamlit as st
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


def _st_image(source, **kwargs):
    """Version-safe wrapper for st.image that handles width params."""
    import inspect
    sig = inspect.signature(st.image)
    if "width" in sig.parameters:
        kwargs.setdefault("width", "stretch")
    elif "use_container_width" in sig.parameters:
        kwargs.setdefault("use_container_width", True)
    elif "use_column_width" in sig.parameters:
        kwargs.setdefault("use_column_width", True)
    st.image(source, **kwargs)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AURA — PCB Inspection Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark professional theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page title (H1) */
    .aura-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.3px;
        margin-bottom: 0.15rem;
        line-height: 1.2;
    }

    /* Subtitle under title */
    .aura-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }

    /* Section headings (H2 equivalent) */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #334155;
    }

    /* Sidebar brand */
    .sidebar-brand {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: 3px;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sidebar-tagline {
        font-size: 0.8rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }

    /* Sidebar section labels */
    .sidebar-section {
        font-size: 1.0rem;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar descriptions */
    .sidebar-description {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .sidebar-description b {
        color: #cbd5e1;
    }

    /* Result card */
    .result-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
        color: #e2e8f0;
        line-height: 1.7;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
    }

    /* Divider */
    .aura-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 1.5rem 0;
    }

    /* Image captions */
    .stImage > div > div > p {
        font-size: 0.8rem;
        color: #94a3b8;
        text-align: center;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-brand">◆ AURA</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-tagline">Augmented Understanding for<br>Reliable Anomaly-detection</div>',
    unsafe_allow_html=True,
)

# ── Manual selection / upload ──────────────────────────────────────────────
st.sidebar.markdown(
    '<div class="sidebar-section">📖 Reference Manual</div>', unsafe_allow_html=True
)

from manual_manager import (
    list_processed_manuals,
    is_manual_processed,
    get_manual_paths,
    process_manual,
    validate_pdf_relevance,
)

_available_manuals = list_processed_manuals()

if _available_manuals:
    _selected_manual = st.sidebar.selectbox(
        "Select manual",
        _available_manuals,
        format_func=lambda mid: get_manual_paths(mid)["manual_name"],
        label_visibility="collapsed",
    )
else:
    _selected_manual = None
    st.sidebar.info("No manuals loaded yet. Upload a PDF below.")

_uploaded_pdf = st.sidebar.file_uploader(
    "Upload a new manual (PDF)", type=["pdf"], key="manual_pdf"
)
if _uploaded_pdf is not None:
    _up_id_candidate = _uploaded_pdf.name
    from manual_manager import _sanitize_id as _sid
    if not is_manual_processed(_sid(_up_id_candidate)):
        _pdf_bytes = _uploaded_pdf.getvalue()
        # Validate relevance before expensive processing
        _is_valid, _reason = validate_pdf_relevance(_pdf_bytes)
        if not _is_valid:
            st.sidebar.error(f"**Rejected:** {_reason}")
        else:
            with st.sidebar:
                st.sidebar.caption(_reason)
                with st.spinner("Processing manual … this may take a few minutes."):
                    _new_id = process_manual(_pdf_bytes, _uploaded_pdf.name)
                st.sidebar.success(f"Manual processed: {_new_id}")
                st.rerun()
    else:
        st.sidebar.info("This manual is already loaded.")

# ── Configure modules for the selected manual ─────────────────────────────
_manual_name_display = "(none)"
if _selected_manual:
    _paths = get_manual_paths(_selected_manual)
    _manual_name_display = _paths["manual_name"]

    from image_analysis_module import configure as _cfg_img
    _cfg_img(
        content_dir=_paths["content_dir"],
        qdrant_dir=_paths["qdrant_dir"],
        storage_dir=_paths["storage_dir"],
        manual_id=_selected_manual,
        manual_name=_manual_name_display,
    )

    from chatbot_module import configure as _cfg_chat
    _cfg_chat(
        pdf_path=_paths["pdf_path"],
        vector_db_dir=_paths["vector_db_dir"],
        cache_dir=_paths["cache_dir"],
        manual_id=_selected_manual,
    )

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<div class="sidebar-section">Mode</div>', unsafe_allow_html=True
)

mode = st.sidebar.radio(
    "Select operating mode",
    ["🔬 Image Analysis", "💬 Manual Chatbot"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-description">'
    "<b>🔬 Image Analysis</b> — Retrieves reference images from the loaded "
    "manual via multimodal RAG and performs defect analysis with an LLM.<br><br>"
    "<b>💬 Chatbot</b> — Answers questions about the loaded manual "
    "using a local Ollama model with retrieval-augmented generation."
    "</div>",
    unsafe_allow_html=True,
)

# =========================================================================
# MODE 1 — Image Analysis
# =========================================================================
if mode == "🔬 Image Analysis":
    st.markdown(
        '<div class="aura-header">🔬 Image Analysis</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="aura-subtitle">'
        f"Upload an image to retrieve matching references from "
        f"<b>{_manual_name_display}</b> and generate an automated defect analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    if not _selected_manual:
        st.warning("Please upload and select a reference manual in the sidebar first.")
        st.stop()

    # Model selector
    from image_analysis_module import AVAILABLE_MODELS, DEFAULT_MODEL

    st.sidebar.markdown(
        '<div class="sidebar-section">Analysis Model</div>', unsafe_allow_html=True
    )
    selected_model = st.sidebar.selectbox(
        "Model",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
        label_visibility="collapsed",
    )

    model_info = AVAILABLE_MODELS[selected_model]
    if model_info["provider"] == "openai" and not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "**OPENAI_API_KEY** not found. "
            "Please set it in the `.env` file in the project root, "
            "or switch to a local Ollama model."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Select a PCB image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()

        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name

        col_upload, col_refs = st.columns([1, 1])
        with col_upload:
            st.markdown(
                '<div class="section-title">Uploaded Image</div>',
                unsafe_allow_html=True,
            )
            _st_image(uploaded_bytes)

        if st.button("▶ Run Analysis", type="primary"):
            with st.spinner(f"Retrieving {_manual_name_display} references..."):
                from image_analysis_module import retrieve_similar_images

                refs = retrieve_similar_images(tmp_path, top_k=4)

            with col_refs:
                st.markdown(
                    f'<div class="section-title">{_manual_name_display} References</div>',
                    unsafe_allow_html=True,
                )
                if refs:
                    ref_cols = st.columns(min(len(refs), 4))
                    for i, rp in enumerate(refs):
                        with ref_cols[i % len(ref_cols)]:
                            if os.path.isfile(rp):
                                from PIL import Image as PILImage

                                ref_img = PILImage.open(rp)
                                _st_image(
                                    ref_img,
                                    caption=Path(rp).stem.replace("_", " "),
                                )
                            else:
                                st.warning(f"Reference not found: {rp}")
                else:
                    st.info("No reference images retrieved.")

            # =============================================================
            # Stage 2 — Defect Analysis Agent
            # =============================================================
            model_label = selected_model.split(" (")[0]
            with st.spinner(f"Defect Analysis Agent ({model_label})..."):
                from image_analysis_module import analyze_image

                analysis, has_defect, refs = analyze_image(
                    tmp_path,
                    reference_paths=refs,
                    model_name=selected_model,
                )

            st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">Defect Analysis</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="result-card">{analysis}</div>',
                unsafe_allow_html=True,
            )

            # If no defect detected the workflow ends here
            if not has_defect:
                st.success("No defect detected — the board appears acceptable.")
            else:
                # =========================================================
                # Stage 3 — Segmentation Detection Agent
                # =========================================================
                st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Defect Segmentation (Sa2VA-4B)</div>',
                    unsafe_allow_html=True,
                )

                seg_text = ""
                try:
                    with st.spinner("Segmentation Detection Agent..."):
                        from image_analysis_module import segment_image

                        seg_text, seg_overlay = segment_image(tmp_path)

                    if seg_overlay is not None:
                        col_orig, col_seg = st.columns([1, 1])
                        with col_orig:
                            st.markdown(
                                '<div class="section-label" style="font-size:0.95rem;font-weight:500;color:#cbd5e1;margin-bottom:0.4rem;">Original</div>',
                                unsafe_allow_html=True,
                            )
                            _st_image(uploaded_bytes)
                        with col_seg:
                            st.markdown(
                                '<div class="section-label" style="font-size:0.95rem;font-weight:500;color:#cbd5e1;margin-bottom:0.4rem;">Segmentation Overlay</div>',
                                unsafe_allow_html=True,
                            )
                            _st_image(seg_overlay)
                    else:
                        st.info("No segmentation masks were produced for this image.")

                    if seg_text:
                        st.markdown(
                            f'<div class="result-card">{seg_text}</div>',
                            unsafe_allow_html=True,
                        )
                except RuntimeError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Segmentation failed: {e}")

                # =========================================================
                # Stage 4 — Explanation Agent
                # =========================================================
                st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="section-title">Explanation &amp; {_manual_name_display} References</div>',
                    unsafe_allow_html=True,
                )

                with st.spinner(f"Explanation Agent ({model_label})..."):
                    from image_analysis_module import explain_defect

                    explanation = explain_defect(
                        defect_analysis=analysis,
                        segmentation_text=seg_text,
                        reference_paths=refs,
                        model_name=selected_model,
                    )

                st.markdown(
                    f'<div class="result-card">{explanation}</div>',
                    unsafe_allow_html=True,
                )

# =========================================================================
# MODE 2 — Manual Chatbot
# =========================================================================
elif mode == "💬 Manual Chatbot":
    st.markdown(
        '<div class="aura-header">💬 Manual Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="aura-subtitle">'
        f"Ask questions about <b>{_manual_name_display}</b>. "
        "Answers are generated locally via Ollama with RAG."
        "</div>",
        unsafe_allow_html=True,
    )

    if not _selected_manual:
        st.warning("Please upload and select a reference manual in the sidebar first.")
        st.stop()

    # Ollama settings in sidebar
    st.sidebar.markdown(
        '<div class="sidebar-section">Ollama Settings</div>', unsafe_allow_html=True
    )

    from chatbot_module import (
        list_available_ollama_models,
        set_ollama_model,
        get_ollama_model,
    )

    available = list_available_ollama_models()
    if available:
        selected_model = st.sidebar.selectbox(
            "Model",
            available,
            index=(
                available.index(get_ollama_model())
                if get_ollama_model() in available
                else 0
            ),
        )
        set_ollama_model(selected_model)
    else:
        custom_model = st.sidebar.text_input(
            "Model Name",
            value=get_ollama_model(),
            help="Ollama is not reachable or has no models. Enter a model name manually.",
        )
        set_ollama_model(custom_model)
        st.sidebar.warning(
            "Cannot reach Ollama at localhost:11434. "
            "Make sure it is running (`ollama serve`)."
        )

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Ask about {_manual_name_display}...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Searching {_manual_name_display} and generating answer..."):
                from chatbot_module import answer_question

                answer = answer_question(user_input)

            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

    if st.session_state.chat_history:
        if st.sidebar.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
