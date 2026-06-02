import re


def extract_roll_number(text, filename):

    match = re.search(
        r'22BCS\d+',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group()

    return filename