"""
Integration tests for the /negotiate endpoint.

These tests use FastAPI's TestClient against the real application, which in
turn calls a real Ollama instance.  They are intentionally NOT mocked — the
goal is to verify end-to-end behaviour including actual LLM responses.

Prerequisites:
  - Ollama running at http://localhost:11434 (or OLLAMA_BASE_URL env var)
  - llama3 model pulled: `ollama pull llama3`

Run with:
  pytest tests/test_negotiation.py -v
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

LOG_FILE = Path(__file__).parent.parent / "negotiation_log.json"


def _post_negotiate(issue: str, rounds: int) -> dict:
    response = client.post("/negotiate", json={"issue": issue, "rounds": rounds})
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text}"
    )
    return response.json()


# ---------------------------------------------------------------------------
# Round count tests
# ---------------------------------------------------------------------------


def test_negotiate_returns_requested_round_count_three():
    """POST with rounds=3 must return exactly 3 round objects."""
    data = _post_negotiate("Tariff reduction on technology goods", rounds=3)
    assert len(data["rounds"]) == 3, (
        f"Expected 3 rounds, got {len(data['rounds'])}"
    )


def test_negotiate_returns_requested_round_count_one():
    """POST with rounds=1 must return exactly 1 round object."""
    data = _post_negotiate("Agricultural market access", rounds=1)
    assert len(data["rounds"]) == 1, (
        f"Expected 1 round, got {len(data['rounds'])}"
    )


# ---------------------------------------------------------------------------
# Round object structure
# ---------------------------------------------------------------------------


def test_round_objects_contain_both_agents():
    """Every round dict must have non-empty string usa_proposal and china_response."""
    data = _post_negotiate("Technology transfer restrictions", rounds=2)
    for i, rnd in enumerate(data["rounds"], start=1):
        assert isinstance(rnd.get("usa_proposal"), str) and rnd["usa_proposal"].strip(), (
            f"Round {i} usa_proposal is missing or empty"
        )
        assert isinstance(rnd.get("china_response"), str) and rnd["china_response"].strip(), (
            f"Round {i} china_response is missing or empty"
        )


# ---------------------------------------------------------------------------
# Outcome schema
# ---------------------------------------------------------------------------


def test_outcome_schema_and_types():
    """outcome must have exactly agreement_reached (bool), final_terms (str), compromise_score (float/int)."""
    data = _post_negotiate("Intellectual property clauses", rounds=1)
    outcome = data["outcome"]

    assert set(outcome.keys()) == {"agreement_reached", "final_terms", "compromise_score"}, (
        f"Unexpected outcome keys: {set(outcome.keys())}"
    )
    assert isinstance(outcome["agreement_reached"], bool), (
        f"agreement_reached should be bool, got {type(outcome['agreement_reached'])}"
    )
    assert isinstance(outcome["final_terms"], str) and outcome["final_terms"].strip(), (
        "final_terms should be a non-empty string"
    )
    assert isinstance(outcome["compromise_score"], (float, int)), (
        f"compromise_score should be float or int, got {type(outcome['compromise_score'])}"
    )


# ---------------------------------------------------------------------------
# Compromise score range
# ---------------------------------------------------------------------------


def test_compromise_score_in_range():
    """compromise_score must be in [0.0, 1.0]."""
    data = _post_negotiate("Trade deficit reduction", rounds=2)
    score = data["outcome"]["compromise_score"]
    assert 0.0 <= score <= 1.0, (
        f"compromise_score {score} is outside [0.0, 1.0]"
    )


# ---------------------------------------------------------------------------
# Priority keyword presence
# ---------------------------------------------------------------------------


def test_proposals_reference_initial_priorities():
    """
    At least one USA round text must reference a USA-priority keyword, and at
    least one China round text must reference a China-priority keyword.

    The LLM will rephrase, so we check a broad set of related terms rather
    than exact strings.  Failure here indicates the prompt is not grounding
    the agent in its country's stated priorities.
    """
    data = _post_negotiate(
        "Comprehensive bilateral trade agreement", rounds=3
    )

    usa_keywords = {"tariff", "ip", "protection", "intellectual property", "technology"}
    china_keywords = {
        "market access",
        "tech transfer",
        "technology transfer",
        "agricultural",
        "agriculture",
    }

    usa_texts = " ".join(
        r["usa_proposal"] for r in data["rounds"]
    ).lower()
    china_texts = " ".join(
        r["china_response"] for r in data["rounds"]
    ).lower()

    usa_match = any(kw in usa_texts for kw in usa_keywords)
    china_match = any(kw in china_texts for kw in china_keywords)

    assert usa_match, (
        f"No USA-priority keyword found in usa_proposals. "
        f"Checked: {usa_keywords}\nText: {usa_texts[:500]}"
    )
    assert china_match, (
        f"No China-priority keyword found in china_responses. "
        f"Checked: {china_keywords}\nText: {china_texts[:500]}"
    )


# ---------------------------------------------------------------------------
# Negotiation log persistence
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def clean_log_file():
    """Delete negotiation_log.json before the test, restore nothing after."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    yield


def test_negotiation_log_created(clean_log_file):
    """
    After a successful /negotiate call, negotiation_log.json must:
      - exist on disk
      - parse as valid JSON
      - be a non-empty list whose last entry matches the request issue
    """
    issue = "Semiconductor export controls"
    _post_negotiate(issue, rounds=1)

    assert LOG_FILE.exists(), "negotiation_log.json was not created"

    with LOG_FILE.open("r", encoding="utf-8") as f:
        log = json.load(f)

    assert isinstance(log, list) and len(log) > 0, (
        "negotiation_log.json is not a non-empty list"
    )
    last_entry = log[-1]
    assert last_entry.get("issue") == issue, (
        f"Last log entry issue '{last_entry.get('issue')}' != '{issue}'"
    )
    assert "rounds" in last_entry and "outcome" in last_entry, (
        f"Log entry missing 'rounds' or 'outcome': {last_entry.keys()}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negotiate_rejects_non_positive_rounds():
    """
    rounds <= 0 must be rejected with HTTP 422 (Pydantic Field gt=0 constraint),
    not silently return HTTP 200 with an empty rounds list.
    """
    response = client.post("/negotiate", json={"issue": "Invalid rounds test", "rounds": 0})
    assert response.status_code == 422, (
        f"Expected 422 for rounds=0, got {response.status_code}"
    )

    response = client.post("/negotiate", json={"issue": "Invalid rounds test", "rounds": -1})
    assert response.status_code == 422, (
        f"Expected 422 for rounds=-1, got {response.status_code}"
    )
