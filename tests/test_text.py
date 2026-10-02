from __future__ import annotations

from sharded_index.text import Tokenizer, normalize_text


def test_normalization_keeps_stop_words() -> None:
    assert normalize_text("What is the Fire?") == "what is the fire"


def test_tokenizer_drops_stop_words(tokenizer: Tokenizer) -> None:
    assert tokenizer("What is the Fire?") == ["fire"]


def test_tokenizer_drops_single_character_tokens(tokenizer: Tokenizer) -> None:
    assert tokenizer("a 1 xy") == ["xy"]


def test_tokenizer_keeps_digits_and_order(tokenizer: Tokenizer) -> None:
    assert tokenizer("Windows10, python 3.13 bug!") == ["windows10", "python", "13", "bug"]
