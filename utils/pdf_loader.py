import fitz

HEADING_MARKER = "[[HDR]] "


def _looks_like_heading(line_text: str, max_size: float, any_bold: bool) -> bool:
    value = (line_text or "").strip()
    if not value:
        return False

    if len(value) > 140:
        return False

    # Strong lexical heading cues in legal documents.
    lower = value.lower()
    heading_starts = (
        "article ",
        "section ",
        "chapter ",
        "clause ",
    )
    if lower.startswith(heading_starts):
        return True

    # Numeric and alphanumeric heading tokens: "1", "1.2", "2.01", "A.", "(a)".
    import re

    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z][\.)]|\([a-zivxlcdm]+\))\b", value):
        return True

    # Style cue: bold or larger text likely indicates heading.
    if any_bold and len(value) <= 120:
        return True
    if max_size >= 11.8 and len(value) <= 100:
        return True

    return False


def _is_noise_line(line_text: str) -> bool:
    value = (line_text or "").strip()
    if not value:
        return True

    import re

    lower = value.lower()
    if "https://www.sec.gov/" in lower or "http://www.sec.gov/" in lower:
        return True
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", lower):
        return True
    if re.search(r"^\d+\s*/\s*\d+\s*$", lower):
        return True
    if re.search(r"^\d+\s+ex-\d", lower):
        return True
    return False


def strip_heading_markers(text: str) -> str:
    return str(text or "").replace(HEADING_MARKER, "")


def extract_text_from_pdf(pdf_path):
    lines = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    parts = []
                    max_size = 0.0
                    any_bold = False
                    for span in spans:
                        text = str(span.get("text", ""))
                        if text:
                            parts.append(text)
                        size = float(span.get("size", 0.0) or 0.0)
                        if size > max_size:
                            max_size = size
                        font_name = str(span.get("font", "")).lower()
                        flags = int(span.get("flags", 0) or 0)
                        if "bold" in font_name or (flags & 16):
                            any_bold = True

                    line_text = "".join(parts).strip()
                    if not line_text:
                        continue
                    if _is_noise_line(line_text):
                        continue

                    if _looks_like_heading(line_text, max_size=max_size, any_bold=any_bold):
                        lines.append(f"{HEADING_MARKER}{line_text}")
                    else:
                        lines.append(line_text)

    return "\n".join(lines).strip()
