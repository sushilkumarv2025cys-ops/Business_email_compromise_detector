"""BECGuard inference: cached model + single-pass compiled indicator matching."""
import re
from email.utils import parseaddr
from functools import lru_cache
from pathlib import Path

import joblib

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "bec_model.joblib"

_RAW = {
    "Urgency": [r"urgent", r"immediately", r"asap", r"right away", r"today",
                r"before the end of the day"],
    "Financial request": [r"transfer", r"payment", r"wire", r"bank account", r"funds",
                          r"beneficiary", r"gift cards?"],
    "Credential request": [r"password", r"credentials?", r"login", r"one[- ]time code",
                           r"authentication code"],
    "Impersonation": [r"ceo", r"cfo", r"chief executive", r"managing director",
                      r"director", r"i am the"],
    "Secrecy": [r"confidential", r"don['’]t discuss", r"do not discuss",
                r"keep this private", r"do not tell"],
    "Process bypass": [r"bypass", r"skip\b.*\bapproval", r"(?:without|no time for)\b.*\bapproval",
                       r"do not verify"],
}
# One compiled regex per category (instead of re-scanning per pattern).
INDICATORS = {k: re.compile(r"\b(?:" + "|".join(v) + r")\b", re.I) for k, v in _RAW.items()}


@lru_cache(maxsize=1)
def get_model():
    """Load the model once per process (lazy, so importing is cheap)."""
    return joblib.load(MODEL_PATH)


def find_indicators(text):
    return [name for name, rx in INDICATORS.items() if rx.search(text)]


def _domain(addr):
    return parseaddr(addr or "")[1].rpartition("@")[2].strip().lower()


def analyze(text, sender="", subject="", reply_to="", external=False):
    text = (text or "").strip()
    subject = (subject or "").strip()
    empty = {"score": 0, "level": "LOW", "probability": 0.0, "indicators": [],
             "metadata_indicators": [], "reasons": []}
    if not text:
        return empty

    # Subject is now part of the analysed content (it was ignored before).
    content = f"{subject}. {text}" if subject else text

    prob = float(get_model().predict_proba([content])[0][1])
    flags = find_indicators(content)

    score = round(prob * 100)
    if not flags and score < 55:
        score = min(score, 24)

    metadata_flags = []
    sd, rd = _domain(sender), _domain(reply_to)
    if sd and rd and sd != rd:
        metadata_flags.append("Sender/Reply-To domain mismatch")
    if external:
        metadata_flags.append("External sender")

    # Metadata alone never makes a message high risk.
    score = min(100, score + 5 * len(metadata_flags))
    if flags:
        floor = min(85, 20 + 10 * len(flags))
        # Keywords alone don't escalate a message the model finds clearly benign.
        if prob < 0.2:
            floor = min(floor, 29)
        score = max(score, floor)

    level = "HIGH" if score >= 70 else "MEDIUM" if score >= 30 else "LOW"

    reasons = [f"Detected {f.lower()} pattern(s)." for f in flags]
    reasons += [f"{f}." for f in metadata_flags]
    if not reasons:
        reasons.append("No predefined BEC indicators were detected.")

    return {"score": score, "level": level, "probability": prob,
            "indicators": flags, "metadata_indicators": metadata_flags,
            "reasons": reasons}
