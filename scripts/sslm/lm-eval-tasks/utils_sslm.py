"""Custom scoring helpers for the SSLM quick-benchmark task variants.

These back task YAMLs whose scoring must tolerate chatty/reasoning-model output.
The built-in chat tasks score 0 for verbose models: arc remove_whitespace cannot
extract a letter from a sentence, and nq_open exact-match never matches a full
sentence against a short entity. All functions read raw dataset fields only, so
they are stable across lm-eval harness versions.
"""
import re

_CHOICES = ["A", "B", "C", "D", "E"]

# First standalone choice letter in a (possibly chatty) completion:
#   "C"  "C."  "C)"  "The answer is C"  "C. Planetary days ..."
_LETTER_RE = re.compile(r"\b([A-E])\b")


def _gold_letter(doc: dict) -> str:
    """Return the gold answer letter for an ARC doc using raw dataset fields."""
    ak = str(doc.get("answerKey", "")).strip()
    if ak.isdigit():
        idx = int(ak) - 1
        return _CHOICES[idx] if 0 <= idx < len(_CHOICES) else ""
    return ak


def arc_process_results(doc: dict, results: list) -> dict:
    """Score ARC by extracting the first choice letter from a free-form answer."""
    completion = (results[0] or "").strip()
    pred = ""
    if completion and completion[0] in _CHOICES:
        pred = completion[0]
    else:
        m = _LETTER_RE.search(completion)
        if m:
            pred = m.group(1)
    return {"exact_match": int(pred == _gold_letter(doc))}


def nq_contains(doc: dict, results: list) -> dict:
    """Score NQ-Open by substring containment of any gold alias.

    Open-domain QA with strict exact-match is unreachable for chatty models that
    answer in full sentences. Containment of a gold alias is a forgiving but
    meaningful signal for a quick smoke suite.
    """
    completion = (results[0] or "").lower()
    golds = doc.get("answer") or []
    if isinstance(golds, str):
        golds = [golds]
    hit = any(g and g.lower() in completion for g in golds)
    return {"contains": int(hit)}
