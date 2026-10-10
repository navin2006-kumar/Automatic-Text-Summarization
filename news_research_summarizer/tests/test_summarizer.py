from types import SimpleNamespace

import pytest

from summarizer import Summarizer


class FakeTensor:
    shape = (1, 12)

    def to(self, device):
        return self


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, **kwargs):
        return " ".join(f"word{i}" for i in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"input_ids": FakeTensor()}


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=64)

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[1, 2, 3]]


def make_fake(name):
    model = Summarizer(name, device="cpu")
    model._tokenizer = FakeTokenizer()
    model._model = FakeModel()
    model._device = "cpu"
    model._context_limit = 64
    return model


@pytest.mark.parametrize("name", ["BART", "T5"])
def test_short_summary_result(name):
    result = make_fake(name).summarize("A short article with useful facts.", 40)
    assert result.model == name
    assert result.summary
    assert result.input_words == 6
    assert result.chunks_processed == 1
    assert result.elapsed_seconds >= 0


def test_reject_invalid_model_and_length():
    with pytest.raises(ValueError, match="Unknown model"):
        Summarizer("other")
    with pytest.raises(ValueError, match="between 20 and 500"):
        make_fake("BART").summarize("a text", 10)


def test_long_input_uses_all_chunks():
    model = make_fake("BART")
    result = model.summarize(" ".join(f"token{i}" for i in range(150)), 40)
    assert result.input_words == 150
    assert result.chunks_processed > 1
