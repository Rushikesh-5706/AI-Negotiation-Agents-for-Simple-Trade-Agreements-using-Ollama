from typing import List, Dict

CONCESSION_KEYWORDS = [
    "agree",
    "accept",
    "concede",
    "compromise",
    "lower",
    "reduce",
    "flexible",
    "willing",
    "acceptable",
    "reasonable",
    "meet halfway",
    "in principle",
]

COMBATIVE_KEYWORDS = [
    "reject",
    "refuse",
    "non-negotiable",
    "unacceptable",
    "won't",
    "will not",
    "cannot accept",
    "must not",
    "increase",
    "firm",
    "no compromise",
]


def calculate_compromise(history: List[Dict], positions: dict) -> float:
    """
    Derive a compromise score in [0.0, 1.0] from the negotiation transcript.

    For each round, the combined text of usa_proposal and china_response is
    scanned for substring matches against CONCESSION_KEYWORDS and
    COMBATIVE_KEYWORDS (case-insensitive).  The score is the fraction of
    total keyword hits that were concession-type hits.

    The `positions` parameter is kept in the signature for interface
    compatibility — future scoring variants may weight keywords by country
    flexibility values.
    """
    concession_total = 0
    combative_total = 0

    for entry in history:
        combined_text = (
            entry.get("usa_proposal", "") + " " + entry.get("china_response", "")
        ).lower()

        for keyword in CONCESSION_KEYWORDS:
            concession_total += combined_text.count(keyword.lower())

        for keyword in COMBATIVE_KEYWORDS:
            combative_total += combined_text.count(keyword.lower())

    raw_score = concession_total / max(1, concession_total + combative_total)

    # Defensive clamp — the formula above cannot exceed [0.0, 1.0] by construction,
    # but an unexpected caller passing fabricated counts could violate this.
    score = round(max(0.0, min(1.0, raw_score)), 2)
    return score
