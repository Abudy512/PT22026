import pytest

from voiceforge.text import (
    normalize_text,
    text_to_sequence,
    sequence_to_text,
    get_unsupported_characters,
)


def test_normalize_text():
    assert normalize_text("  HELLO   WORLD!  ") == "hello world!"


def test_text_to_sequence():
    assert text_to_sequence("cat") == [3, 1, 20]


def test_sequence_to_text():
    assert sequence_to_text([3, 1, 20]) == "cat"


def test_get_unsupported_characters():
    assert get_unsupported_characters("hello 123") == ["1", "2", "3"]


def test_unsupported_characters_raise_error():
    with pytest.raises(ValueError):
        text_to_sequence("hello 123")


def test_invalid_character_id_raises_error():
    with pytest.raises(ValueError):
        sequence_to_text([3, 999])