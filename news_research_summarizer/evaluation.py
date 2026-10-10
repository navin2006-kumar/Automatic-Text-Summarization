"""Reproducible paired BART/T5 evaluation against reference summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from summarizer import SummaryResult, Summarizer


REQUIRED_COLUMNS = {"text", "reference"}


def compute_rouge(
    predictions: list[str], references: list[str], metric: Any | None = None
) -> dict[str, float]:
    """Compute corpus ROUGE with Hugging Face Evaluate (ROUGE is overlap, not truth)."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must contain the same number of items.")
    if not predictions:
        raise ValueError("At least one prediction/reference pair is required.")
    if any(not p.strip() for p in predictions) or any(not r.strip() for r in references):
        raise ValueError("Predictions and references must not be empty.")
    if metric is None:
        import evaluate

        metric = evaluate.load("rouge")
    raw = metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {key: float(raw[key]) for key in ("rouge1", "rouge2", "rougeL")}


def evaluate_frame(
    frame: pd.DataFrame,
    summarizer_factory: Callable[[str], Any] = Summarizer,
    max_summary_words: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both models over the exact same input rows and references."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {', '.join(sorted(missing))}.")
    samples = frame.loc[:, ["text", "reference"]].dropna()
    samples = samples[
        samples["text"].astype(str).str.strip().ne("")
        & samples["reference"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    if samples.empty:
        raise ValueError("No non-empty text/reference pairs were found.")

    rows: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for model_name in ("BART", "T5"):
        model = summarizer_factory(model_name)
        predictions: list[str] = []
        model_rows: list[dict[str, Any]] = []
        for index, sample in samples.iterrows():
            result: SummaryResult = model.summarize(
                str(sample["text"]), max_summary_words=max_summary_words
            )
            predictions.append(result.summary)
            model_rows.append(
                {
                    "sample_id": index + 1,
                    "model": model_name,
                    "reference": str(sample["reference"]),
                    "summary": result.summary,
                    "input_words": result.input_words,
                    "summary_words": result.summary_words,
                    "compression_ratio": result.compression_ratio,
                    "inference_seconds": result.elapsed_seconds,
                    "chunks_processed": result.chunks_processed,
                }
            )
        metrics = compute_rouge(
            predictions, [str(reference) for reference in samples["reference"]]
        )
        for row in model_rows:
            row.update(metrics)
        rows.extend(model_rows)
        aggregate.append(
            {
                "model": model_name,
                "sample_count": len(model_rows),
                "rouge1": metrics["rouge1"],
                "rouge2": metrics["rouge2"],
                "rougeL": metrics["rougeL"],
                "average_inference_seconds": sum(
                    row["inference_seconds"] for row in model_rows
                )
                / len(model_rows),
                "average_compression_ratio": sum(
                    row["compression_ratio"] for row in model_rows
                )
                / len(model_rows),
                "average_summary_words": sum(
                    row["summary_words"] for row in model_rows
                )
                / len(model_rows),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with text and reference columns")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-summary-words", type=int, default=150)
    args = parser.parse_args()

    input_path = Path(args.input)
    frame = pd.read_csv(input_path)
    details, summary = evaluate_frame(frame, max_summary_words=args.max_summary_words)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "evaluation_details.csv", index=False)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    metadata = {
        "input_file": str(input_path.resolve()),
        "sample_count": int(summary["sample_count"].iloc[0]),
        "models": ["facebook/bart-large-cnn", "t5-base"],
        "note": "ROUGE is lexical overlap and does not establish factual correctness.",
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nDetailed results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
