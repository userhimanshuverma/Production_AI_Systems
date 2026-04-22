"""
Text cleaning: removes boilerplate, normalizes whitespace,
filters near-empty content.
"""
import re


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove lines that are clearly boilerplate
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 15:
            continue  # too short — page numbers, headers, etc.
        if re.match(r"^[©\-_=\*]{3,}$", stripped):
            continue  # decorative lines
        if "cookie" in stripped.lower() and len(stripped) < 80:
            continue  # cookie notices
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
