import pandas as pd
import pytest

from evaluation import compute_rouge, evaluate_frame
from summarizer import SummaryResult


class FakeRouge:
    def compute(self, predictions, references, use_stemmer):
        assert len(predictions) == len(references) == 1
        assert use_stemmer is True
        return {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.4}


class FakeSummarizer:
    def __init__(self, name):
        self.name = name

    def summarize(self, text, max_summary_words):
        return SummaryResult(
            summary=f"{self.name} summary",
            model=self.name,
            model_id="test/model",
            elapsed_seconds=0.1,
            input_words=len(text.split()),
            summary_words=2,
            compression_ratio=2 / len(text.split()),
            chunks_processed=1,
        )


def test_rouge_validation_and_metric_values():
    assert compute_rouge(["summary"], ["reference"], FakeRouge()) == {
        "rouge1": 0.5,
        "rouge2": 0.25,
        "rougeL": 0.4,
    }
    with pytest.raises(ValueError, match="same number"):
        compute_rouge(["one"], [], FakeRouge())


def test_paired_evaluation_uses_same_rows():
    frame = pd.DataFrame(
        {"text": ["source text"], "reference": ["human summary"]}
    )
    details, aggregate = evaluate_frame(frame, FakeSummarizer)
    assert details["model"].tolist() == ["BART", "T5"]
    assert details["sample_id"].tolist() == [1, 1]
    assert aggregate["sample_count"].tolist() == [1, 1]


def test_evaluation_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_frame(pd.DataFrame({"text": ["source"]}), FakeSummarizer)
