"""Streamlit interface for local BART/T5 news and research summarization."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation import compute_rouge
from preprocessing import DocumentProcessingError, extract_text, validate_text, word_count
from summarizer import Summarizer


st.set_page_config(page_title="News & Research Summarizer", page_icon="📝", layout="wide")


@st.cache_resource
def get_summarizer(model_name: str) -> Summarizer:
    model = Summarizer(model_name)  # type: ignore[arg-type]
    model.load()
    return model


def _model_summary(text: str, model_name: str, max_words: int):
    model = get_summarizer(model_name)
    return model.summarize(text, max_summary_words=max_words)


def _show_result(result) -> None:
    st.subheader(f"{result.model} summary")
    st.write(result.summary)
    columns = st.columns(4)
    columns[0].metric("Input words", result.input_words)
    columns[1].metric("Summary words", result.summary_words)
    columns[2].metric("Summary / input", f"{result.compression_ratio:.1%}")
    columns[3].metric("Inference time", f"{result.elapsed_seconds:.2f} s")
    st.caption(
        f"Model: `{result.model_id}` · Input chunks processed: "
        f"{result.chunks_processed}. Runtime excludes initial model download/loading."
    )
    st.download_button(
        "Download summary",
        data=result.summary,
        file_name=f"{result.model.lower()}_summary.txt",
        mime="text/plain",
    )


st.title("Automatic Summarization of News and Research Papers")
st.write(
    "Generate concise abstractive summaries with locally run, pretrained BART and T5 "
    "models. Long documents are split by model tokens and reduced hierarchically."
)
page = st.sidebar.radio(
    "Application page", ["Summarize", "Compare models", "ROUGE evaluation"]
)

if page == "Summarize":
    uploaded = st.file_uploader("Upload TXT, PDF, or DOCX", type=["txt", "pdf", "docx"])
    pasted_text = st.text_area("Or paste article / research text", height=260)
    source = pasted_text
    if uploaded is not None:
        try:
            source = extract_text(uploaded.name, uploaded.getvalue())
            st.success(f"Read {uploaded.name} successfully.")
        except DocumentProcessingError as exc:
            st.error(str(exc))
            source = ""

    if source.strip():
        st.caption(f"Original word count: {word_count(source)}")
        with st.expander("View source text"):
            st.text(source)
    model_name = st.selectbox("Summarization model", ["BART", "T5"])
    max_words = st.slider("Target summary length (words)", 20, 500, 150, 10)
    if st.button("Summarize", type="primary"):
        try:
            cleaned = validate_text(source)
            with st.spinner(f"Loading/running {model_name}; first load may download model weights..."):
                result = _model_summary(cleaned, model_name, max_words)
            _show_result(result)
        except (DocumentProcessingError, ValueError, RuntimeError, OSError, ImportError) as exc:
            st.error(f"Summarization failed: {exc}")

elif page == "Compare models":
    st.subheader("Paired BART and T5 comparison")
    compare_text = st.text_area("Article or research-paper text", height=280)
    compare_words = st.slider("Target summary length (words)", 20, 500, 150, 10, key="compare_words")
    if st.button("Run both models", type="primary"):
        try:
            cleaned = validate_text(compare_text)
            results = []
            for name in ("BART", "T5"):
                with st.spinner(f"Running {name}..."):
                    results.append(_model_summary(cleaned, name, compare_words))
            left, right = st.columns(2)
            with left:
                _show_result(results[0])
            with right:
                _show_result(results[1])
        except (DocumentProcessingError, ValueError, RuntimeError, OSError, ImportError) as exc:
            st.error(f"Comparison failed: {exc}")

elif page == "ROUGE evaluation":
    st.subheader("Reference-based ROUGE evaluation")
    st.warning(
        "ROUGE measures n-gram/sequence overlap with a human reference. It does not "
        "verify factual correctness, faithfulness, or usefulness."
    )
    evaluation_text = st.text_area("Source text", height=220, key="rouge_source")
    reference = st.text_area("Human-written reference summary", height=140)
    eval_words = st.slider("Target summary length (words)", 20, 500, 150, 10, key="rouge_words")
    if st.button("Generate and score both models", type="primary"):
        try:
            cleaned = validate_text(evaluation_text)
            if not reference.strip():
                raise ValueError("Enter a non-empty human reference summary.")
            rows = []
            for name in ("BART", "T5"):
                with st.spinner(f"Generating and scoring {name}..."):
                    result = _model_summary(cleaned, name, eval_words)
                    scores = compute_rouge([result.summary], [reference.strip()])
                rows.append(
                    {
                        "Model": name,
                        "ROUGE-1": scores["rouge1"],
                        "ROUGE-2": scores["rouge2"],
                        "ROUGE-L": scores["rougeL"],
                        "Inference seconds": result.elapsed_seconds,
                        "Summary words": result.summary_words,
                        "Compression ratio": result.compression_ratio,
                    }
                )
                with st.expander(f"{name} generated summary"):
                    st.write(result.summary)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except (DocumentProcessingError, ValueError, RuntimeError, OSError, ImportError) as exc:
            st.error(f"Evaluation failed: {exc}")

st.divider()
st.caption(
    "Chunking can weaken cross-section context and entity resolution. Review summaries "
    "against the source, especially for findings, numbers, and technical claims."
)
