"""Local BART/T5 summarization with token-aware map-reduce chunking."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

from preprocessing import validate_text, word_count


ModelName = Literal["BART", "T5"]
MODEL_IDS: dict[str, str] = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-base",
}
DEFAULT_CONTEXT_LIMITS: dict[str, int] = {"BART": 1024, "T5": 512}


class SummarizationError(RuntimeError):
    """Raised when a model cannot produce a safe, complete-document summary."""


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    model: str
    model_id: str
    elapsed_seconds: float
    input_words: int
    summary_words: int
    compression_ratio: float
    chunks_processed: int


class Summarizer:
    """Load pretrained models on demand and summarize with a full-input pipeline."""

    def __init__(self, model_name: ModelName, device: str | None = None) -> None:
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unknown model '{model_name}'. Choose BART or T5.")
        self.model_name = model_name
        self.model_id = MODEL_IDS[model_name]
        self._requested_device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._context_limit = DEFAULT_CONTEXT_LIMITS[model_name]
        self._chunks_processed = 0

    def load(self) -> None:
        """Load model weights and tokenizer; download errors are surfaced to caller."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = torch.device(
            self._requested_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._model.to(self._device)
        self._model.eval()
        configured_limit = getattr(
            self._model.config,
            "max_position_embeddings",
            getattr(self._model.config, "n_positions", None),
        )
        if isinstance(configured_limit, int) and configured_limit > 0:
            self._context_limit = min(self._context_limit, configured_limit)

    def _token_ids(self, text: str) -> list[int]:
        self.load()
        assert self._tokenizer is not None
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _capacity(self) -> int:
        self.load()
        assert self._tokenizer is not None
        special_tokens = self._tokenizer.num_special_tokens_to_add(pair=False)
        prefix_tokens = (
            len(self._tokenizer.encode("summarize: ", add_special_tokens=False))
            if self.model_name == "T5"
            else 0
        )
        capacity = self._context_limit - special_tokens - prefix_tokens
        if capacity < 32:
            raise SummarizationError(
                f"{self.model_id} exposes an unusably small input-token limit."
            )
        return capacity

    def _split_ids(self, text: str) -> list[str]:
        ids = self._token_ids(text)
        capacity = self._capacity()
        assert self._tokenizer is not None
        return [
            self._tokenizer.decode(
                ids[offset : offset + capacity],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for offset in range(0, len(ids), capacity)
        ]

    def _generate(self, text: str, max_new_tokens: int) -> str:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        prompt = f"summarize: {text}" if self.model_name == "T5" else text
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = int(inputs["input_ids"].shape[-1])
        if input_length > self._context_limit:
            raise SummarizationError(
                f"Internal chunk exceeded the {self._context_limit}-token context limit."
            )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=max(8, max_new_tokens),
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        generated = self._tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        if not generated:
            raise SummarizationError(f"{self.model_id} returned an empty summary.")
        return generated

    def summarize(self, text: str, max_summary_words: int = 150) -> SummaryResult:
        """Summarize all input tokens; recursively reduce chunk summaries if needed."""
        source = validate_text(text)
        if max_summary_words < 20 or max_summary_words > 500:
            raise ValueError("Summary length must be between 20 and 500 words.")

        started = time.perf_counter()
        self.load()
        self._chunks_processed = 0
        parts = self._split_ids(source)
        self._chunks_processed = sum(bool(part) for part in parts)
        if len(parts) == 1:
            condensed = parts[0]
        else:
            condensed = "\n".join(
                self._generate(part, max(16, min(256, math.ceil(len(self._token_ids(part)) * 0.45))))
                for part in parts
                if part
            )

        capacity = self._capacity()
        for _ in range(8):
            condensed_ids = self._token_ids(condensed)
            if len(condensed_ids) <= capacity:
                break
            reduce_parts = self._split_ids(condensed)
            reduced_parts = [
                self._generate(
                    part,
                    max(16, min(256, math.ceil(len(self._token_ids(part)) * 0.40))),
                )
                for part in reduce_parts
                if part
            ]
            reduced = "\n".join(reduced_parts)
            if len(self._token_ids(reduced)) >= len(condensed_ids):
                raise SummarizationError(
                    "Intermediate summaries did not shrink enough to fit the model "
                    "context. No source chunks were dropped; try a shorter document."
                )
            condensed = reduced
        else:
            raise SummarizationError(
                "The hierarchical reduction exceeded eight passes. No source chunks "
                "were dropped; reduce document size or choose a different model."
            )

        final = self._generate(
            condensed,
            max_new_tokens=min(768, math.ceil(max_summary_words * 1.6)),
        )
        output_words = word_count(final)
        input_words = word_count(source)
        return SummaryResult(
            summary=final,
            model=self.model_name,
            model_id=self.model_id,
            elapsed_seconds=time.perf_counter() - started,
            input_words=input_words,
            summary_words=output_words,
            compression_ratio=(output_words / input_words) if input_words else 0.0,
            chunks_processed=self._chunks_processed,
        )
