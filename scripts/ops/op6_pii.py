"""Step 3.6 — mask e-mail addresses and phone numbers (regex based).

Precision is preferred over recall: dates, years and money amounts must not be masked.
Names and street addresses are NOT handled here (that needs NER and human review).
"""
import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_INTL = re.compile(r"(?<![\w+])(?:\+|00)\d[\d \-()]{7,17}\d(?!\w)")
_LOCAL = re.compile(r"(?<![\w.,/:\-+])0\d{1,3}[ \-]?\d{2,4}[ \-]?\d{3,4}(?![\w.,/:\-]*\d)")
_RUN = re.compile(r"(?<![\w.,/:\-+])\d{9,12}(?![\w.,/:\-]*\d)")


def _digits(s):
    return sum(c.isdigit() for c in s)


def mask_pii(text):
    """Return (masked_text, n_emails, n_phones)."""
    text, n_mail = _EMAIL.subn("[EMAIL]", text)
    n_phone = 0

    def repl(m):
        nonlocal n_phone
        d = _digits(m.group(0))
        if 9 <= d <= 15:
            n_phone += 1
            return "[PHONE]"
        return m.group(0)

    text = _INTL.sub(repl, text)
    text = _LOCAL.sub(repl, text)
    text = _RUN.sub(repl, text)
    return text, n_mail, n_phone
