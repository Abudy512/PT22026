
import pytest

from voiceforge.text import (
    CHARACTERS,
    normalize_text,
    text_to_sequence,
    sequence_to_text,
    get_unsupported_characters,
)


def test_normalize_text():
    assert normalize_text("  HELLO   WORLD!  ") == "hello world!"


def test_text_to_sequence_and_back():
    text = "Hello world."

    sequence = text_to_sequence(text)

    assert sequence_to_text(sequence) == "hello world."


def test_unsupported_characters():
    assert get_unsupported_characters("Hello @world") == ["@"]


def test_invalid_character_id():
    with pytest.raises(ValueError, match="Invalid character ID"):
        sequence_to_text([9999])


def test_text_supports_double_quotes():
    text = 'He said "hello".'

    sequence = text_to_sequence(text)

    assert sequence_to_text(sequence) == text.lower()


def test_text_supports_dataset_punctuation():
    text = "Wait; really: (yes)."

    sequence = text_to_sequence(text)

    assert sequence_to_text(sequence) == text.lower()


def test_character_vocabulary_has_no_duplicates():
    assert len(CHARACTERS) == len(set(CHARACTERS))



def test_normalize_text_converts_accented_characters():
    assert normalize_text("à â è é ê ü") == "a a e e e u"


def test_normalize_text_converts_smart_quotes():
    assert normalize_text("“Hello,” she said. It’s fine.") == (
        '"hello," she said. it\'s fine.'
    )


def test_normalize_text_removes_square_brackets():
    assert normalize_text("Hello [marina] world") == "hello marina world"


def test_ljspeech_example_text_is_supported():
    text = (
        "at müller's execution, "
        "there was a three hours’ rise."
    )

    sequence = text_to_sequence(text)

    assert sequence_to_text(sequence) == (
        "at muller's execution, "
        "there was a three hours' rise."
    )
