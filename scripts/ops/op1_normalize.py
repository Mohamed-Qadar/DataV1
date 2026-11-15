"""Step 3.1 — UTF-8 / Unicode NFC normalisation and mojibake repair (ftfy)."""
import re
import unicodedata

import ftfy
from ftfy import badness

_INVISIBLE = re.compile("[​‌‍⁠﻿­]")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACES = re.compile("[ \t  -  　]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize(text):
    """Return cleaned text, or None when the encoding damage cannot be repaired."""
    if "�" in text or badness.is_bad(text):
        text = ftfy.fix_text(text, uncurl_quotes=False, fix_latin_ligatures=False,
                             fix_character_width=False, normalization=None)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text).strip()
    bad = text.count("�")
    if bad and bad / max(1, len(text)) > 0.001:
        return None
    return text.replace("�", "") if bad else text
