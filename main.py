import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.negotiator import Negotiator
from scoring import calculate_compromise

# ---------------------------------------------------------------------------
# Startup: load positions from file relative to this module, not CWD.
# This ensures the path resolves identically whether the process is started
# from the project root locally or from /app inside the Docker container.
# ---------------------------------------------------------------------------
_DATA_FILE = Path(__file__).parent / "data" / "trade_positions.json"

with _DATA_FILE.open("r", encoding="utf-8") as _f:
    positions: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ollama_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Agent instantiation (shared across requests — stateless per-call)
# ---------------------------------------------------------------------------
usa_agent = Negotiator("usa", positions["usa"], ollama_url)
china_agent = Negotiator("china", positions["china"], ollama_url)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Negotiation Agents — Trade Agreements",
    description=(
        "Multi-agent negotiation system where USA and China LLM agents "
        "conduct bilateral trade negotiations via Ollama/Llama 3."
    ),
    version="1.0.0",
)

_LOG_FILE = Path(__file__).parent / "negotiation_log.json"


class NegotiateRequest(BaseModel):
    issue: str
    rounds: int = 3


@app.post("/negotiate")
async def negotiate(request: NegotiateRequest) -> dict:
    """
    Run `request.rounds` rounds of bilateral negotiation on `request.issue`.

    Each round:
      1. USA agent makes a proposal given the current history.
      2. China agent responds given the history including USA's proposal.
      3. The full round dict is appended to the history.

    Returns the complete round history and an outcome summary including the
    compromise score, whether agreement was reached, and final terms.
    """
    history: list[dict] = []

    try:
        for i in range(1, request.rounds + 1):
            usa_proposal = await usa_agent.make_proposal(request.issue, history)

            # China sees the history INCLUDING the current USA proposal so it
            # can respond to it directly.
            china_response = await china_agent.make_proposal(
                request.issue,
                history + [{"round": i, "usa_proposal": usa_proposal}],
            )

            history.append(
                {
                    "round": i,
                    "usa_proposal": usa_proposal,
                    "china_response": china_response,
                }
            )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama LLM service unavailable: {exc}",
        ) from exc

    score = calculate_compromise(history, positions)
    agreement_reached: bool = score > 0.6

    if agreement_reached:
        final_terms = (
            f"Agreement reached on {request.issue} "
            f"with a compromise score of {score}."
        )
    else:
        final_terms = (
            f"No agreement reached on {request.issue}; "
            f"negotiations ended with a compromise score of {score}."
        )

    outcome = {
        "agreement_reached": agreement_reached,
        "final_terms": final_terms,
        "compromise_score": score,
    }

    _append_log(
        {
            "issue": request.issue,
            "rounds": history,
            "outcome": outcome,
        }
    )

    return {"rounds": history, "outcome": outcome}


def _append_log(entry: dict) -> None:
    """
    Append a negotiation entry to negotiation_log.json.

    Reads the existing list if the file is present and contains valid JSON;
    starts a fresh list otherwise.  Writes atomically by replacing the file.
    """
    log: list = []

    if _LOG_FILE.exists():
        try:
            with _LOG_FILE.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                log = existing
        except (json.JSONDecodeError, OSError):
            # File is corrupt or unreadable — start fresh rather than crashing.
            log = []

    log.append(entry)

    with _LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
