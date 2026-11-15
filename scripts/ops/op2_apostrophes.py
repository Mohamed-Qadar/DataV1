"""Step 3.2 — map every apostrophe variant to the plain ' character (U+0027)."""

_TABLE = {ord(c): "'" for c in "’‘‛`´ʼʻʹ′＇՚"}


def fix_apostrophes(text):
    return text.translate(_TABLE)
