import asyncio
import re
import httpx
from typing import List, Dict


class Negotiator:
    """
    Represents a single country's negotiation agent.

    Wraps Ollama LLM calls with retry/backoff logic and constructs
    chain-of-thought prompts that instruct the model to output only
    the final proposal text, hiding internal reasoning.
    """

    def __init__(self, country: str, positions: dict, ollama_url: str) -> None:
        self.country = country
        self.positions = positions
        self.ollama_url = ollama_url
        self.model = "llama3"

    async def generate_response(self, prompt: str) -> str:
        """
        POST prompt to Ollama /api/generate and return the stripped response text.

        Retries up to 3 times on TimeoutException or ConnectError with
        exponential backoff (1 s, 2 s, 4 s).  Raises a RuntimeError naming
        the country and attempt count on final failure so callers can surface
        a clean error rather than receiving fabricated text.
        """
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        backoff_seconds = [1, 2, 4]
        last_exception: Exception | None = None

        for attempt, wait in enumerate(backoff_seconds, start=1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    raw = data.get("response", "")
                    return self._clean_response(raw)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exception = exc
                if attempt < len(backoff_seconds):
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama returned HTTP {exc.response.status_code} for "
                    f"{self.country} (attempt {attempt}): {exc.response.text}"
                ) from exc

        raise RuntimeError(
            f"Ollama unreachable for country '{self.country}' after "
            f"{len(backoff_seconds)} attempts: {last_exception}"
        )

    def _clean_response(self, text: str) -> str:
        """
        Strip artefacts that the model may add despite instructions:
        - Leading/trailing whitespace
        - Surrounding quote characters
        - Markdown code fences (``` blocks)
        - Common preamble labels the model ignores formatting for
        """
        text = text.strip()

        # Remove wrapping quotes (single or double)
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
            text = text[1:-1].strip()

        # Strip markdown code fences
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)
        text = text.strip()

        # Strip common preamble labels (case-insensitive)
        preamble_pattern = re.compile(
            r"^(here\s+is\s+my\s+proposal\s*:|proposal\s*:|response\s*:)\s*",
            flags=re.IGNORECASE,
        )
        text = preamble_pattern.sub("", text).strip()

        return text

    async def make_proposal(self, issue: str, history: List[Dict]) -> str:
        """
        Build a chain-of-thought prompt and return the model's final proposal.

        The prompt tells the model to reason internally but emit ONLY the
        final 1-2 sentence proposal — no labels, no visible reasoning chain.
        """
        priorities_text = "\n".join(
            f"  - {p}" for p in self.positions["priorities"]
        )
        flexibility_text = "\n".join(
            f"  - {k}: {v}" for k, v in self.positions["flexibility"].items()
        )

        if history:
            history_lines = []
            for entry in history:
                history_lines.append(
                    f"  Round {entry['round']}: "
                    f"USA proposed: {entry.get('usa_proposal', '(pending)')} | "
                    f"China responded: {entry.get('china_response', '(pending)')}"
                )
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (no prior rounds)"

        prompt = (
            f"You are a senior trade negotiator representing {self.country.upper()} "
            f"in a bilateral trade negotiation on the following issue:\n\n"
            f"Issue: {issue}\n\n"
            f"Your country's priorities are:\n{priorities_text}\n\n"
            f"Your flexibility scores (0 = rigid, 1 = highly flexible):\n"
            f"{flexibility_text}\n\n"
            f"Negotiation history so far:\n{history_text}\n\n"
            f"Instructions:\n"
            f"1. Think through your negotiation strategy, considering your priorities, "
            f"flexibility, and the history above.\n"
            f"2. Decide what concessions, if any, to offer, and what you need in return.\n"
            f"3. Your proposal MUST explicitly reference at least one of your stated "
            f"priorities by name (e.g. mention 'tariffs', 'IP protection', "
            f"'market access', or 'technology transfer' directly).\n"
            f"4. Output ONLY your final proposal as 1-2 sentences. "
            f"Do NOT include labels like 'Proposal:', 'Response:', reasoning steps, "
            f"or any preamble. Begin directly with the proposal text.\n"
        )

        return await self.generate_response(prompt)
