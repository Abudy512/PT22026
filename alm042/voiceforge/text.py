import re


def normalize_text(text: str) -> str:
    """
    Clean and standardize input text.
    """

    # Remove spaces at the beginning and end
    text = text.strip()

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # Convert text to lowercase
    text = text.lower()

    return text

# Characters supported by VoiceForge
CHARACTERS = " abcdefghijklmnopqrstuvwxyz'.,!?-"

# Create a dictionary that assigns an ID to each character
CHAR_TO_ID = {
    char: index
    for index, char in enumerate(CHARACTERS)
}

def text_to_sequence(text: str) -> list[int]:
    """
    Convert text into a sequence of character IDs.
    """

    text = normalize_text(text)

    unsupported = get_unsupported_characters(text)

    if unsupported:
        raise ValueError(
            f"Unsupported characters: {unsupported}"
        )

    sequence = [
        CHAR_TO_ID[char]
        for char in text
    ]

    return sequence
def sequence_to_text(sequence: list[int]) -> str:
    """
    Convert a sequence of character IDs back into text.
    """

    id_to_char = {
        index: char
        for index, char in enumerate(CHARACTERS)
    }

    for index in sequence:
        if index not in id_to_char:
            raise ValueError(f"Invalid character ID: {index}")

    text = "".join(
        id_to_char[index]
        for index in sequence
    )

    return text
def get_unsupported_characters(text: str) -> list[str]:
    """
    Find characters that are not supported by VoiceForge.
    """

    text = normalize_text(text)

    unsupported = [
        char for char in text
        if char not in CHAR_TO_ID
    ]

    return sorted(set(unsupported))