
"""
Text processing utilities for VoiceForge.
"""

import re


CHARACTERS = " abcdefghijklmnopqrstuvwxyz'.,!?-\";:"

CHAR_TO_ID = {
    character: index
    for index, character in enumerate(CHARACTERS)
}


def normalize_text(text: str) -> str:
    """Normalize text before converting it into character IDs."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def text_to_sequence(text: str) -> list[int]:
    """Convert text into a sequence of character IDs."""
    text = normalize_text(text)

    unsupported = get_unsupported_characters(text)

    if unsupported:
        raise ValueError(
            f"Unsupported characters: {unsupported}"
        )

    return [CHAR_TO_ID[character] for character in text]


def sequence_to_text(sequence: list[int]) -> str:
    """Convert character IDs back into text."""
    id_to_char = {
        index: character
        for character, index in CHAR_TO_ID.items()
    }

    characters = []

    for index in sequence:
        if index not in id_to_char:
            raise ValueError(
                f"Invalid character ID: {index}"
            )

        characters.append(id_to_char[index])

    return "".join(characters)


def get_unsupported_characters(text: str) -> list[str]:
    """Return unique characters that are not in our vocabulary."""
    text = normalize_text(text)

    return sorted({
        character
        for character in text
        if character not in CHAR_TO_ID
    })
