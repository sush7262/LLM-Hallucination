"""
Stage 4 — Aggregation

Converts per-claim verdicts into a single answer-level decision.

Rule (validated, matches report Section 4.3):
  • Any claim contradicted   →  REJECT / REGENERATE
  • Else, any unsupported    →  PASS WITH FLAGS
  • Else (all supported)     →  PASS

Known limitation: this is a worst-case rule — one misclassified claim
can tank the whole answer. This is a design trade-off, not a bug.
See PROJECT_SPEC.md Section 3, Stage 4.
"""


def aggregate(labels: list[str]) -> str:
    """
    Determine the answer-level decision from a list of per-claim labels.

    Args:
        labels: List of claim labels, each ∈ {supported, unsupported, contradicted}.

    Returns:
        One of the three decision strings.
    """
    if not labels:
        return "PASS (fully supported)"

    if "contradicted" in labels:
        return "REJECT / REGENERATE (contradiction found)"

    if "unsupported" in labels:
        return "PASS WITH FLAGS (some claims unverifiable)"

    return "PASS (fully supported)"
