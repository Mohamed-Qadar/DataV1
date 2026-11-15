"""Step 3.4 — drop documents under 50 words, plus a conservative noise filter.

The noise filter only removes clear junk (tables of numbers, shouting spam, menu
repetition, gambling/pharma spam). Short-line texts such as poetry are not touched.
"""
import re

MIN_WORDS = 50
_SPAM = re.compile(r"\b(casino|poker|bet365|1xbet|betting|slots?|viagra|cialis|porn|xxx|escort|"
                   r"steroids?|clenbuterol|anavar|dianabol|forex signals|lorem ipsum)\b", re.I)


def too_short(n_words):
    return n_words < MIN_WORDS


def is_noise(text):
    n = len(text)
    letters = sum(c.isalpha() for c in text)
    if letters / n < 0.60:
        return True
    upper = sum(c.isupper() for c in text)
    if upper / max(1, letters) > 0.30:
        return True
    lines = [l for l in text.split("\n") if l]
    if len(lines) >= 8 and 1 - len(set(lines)) / len(lines) > 0.30:
        return True
    if len(_SPAM.findall(text)) >= 3:
        return True
    return False
