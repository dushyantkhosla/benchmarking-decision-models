"""Model adapter for decision models served by ollaya (https://ollaya.dev).

Every model adapter exposes the same conceptual operation:
    decide(example) -> prediction
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from core.schemas import DecisionExample, Prediction


@dataclass
class OllayaConfig:
    host: str = "http://localhost:11435"
    keep_alive: str = "10m"
    timeout_s: float = 180.0


class OllayaDecisionModel:
    def __init__(self, model: str, config: OllayaConfig | None = None):
        self.model = model
        self.config = config or OllayaConfig()
        self._client = httpx.Client(base_url=self.config.host, timeout=self.config.timeout_s)

    def version(self) -> str:
        return self._client.get("/api/version").json()["version"]

    def show(self) -> dict:
        return self._client.post("/api/show", json={"model": self.model}).json()

    def warmup(self) -> None:
        """Load the model and keep it resident; ignore errors."""
        try:
            self._client.post(
                "/api/decide", json={"model": self.model, "keep_alive": -1}
            )
        except Exception:
            pass

    def decide(self, example: DecisionExample) -> Prediction:
        decision = example.decision
        dtype = decision.get("type")
        if dtype not in ("choice", "noul"):
            raise ValueError(f"unsupported decision type: {dtype}")

        question = decision.get("question", "decision")
        if dtype == "choice":
            options = decision["options"]
            question_spec = {
                "type": "choice",
                "instructions": decision.get("instructions", "Choose the single best option."),
                "criteria": {opt: opt for opt in options},
            }
        else:  # noul: P(statement holds)
            options = ["yes", "no"]
            question_spec = {
                "type": "noul",
                "instructions": decision.get("instructions", "Does the statement hold?"),
            }

        payload = {
            "model": self.model,
            "state": example.input["text"],
            "questions": {question: question_spec},
            "keep_alive": self.config.keep_alive,
        }

        t0 = time.perf_counter()
        try:
            resp = self._client.post("/api/decide", json=payload)
            wall_ms = (time.perf_counter() - t0) * 1000
            body = resp.json()
        except Exception as exc:  # network/timeout/decode failure
            return Prediction(
                example_id=example.id, model=self.model, answer=None,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"request_failed: {exc}",
            )

        if resp.status_code != 200:
            return Prediction(
                example_id=example.id, model=self.model, answer=None,
                latency_ms=wall_ms,
                error=f"http_{resp.status_code}: {body.get('error', body)}",
                raw=body if isinstance(body, dict) else None,
            )

        answer = body.get("answers", {}).get(question, {})
        error = None
        if dtype == "choice":
            choice = answer.get("choice")
            if choice not in options:
                error = f"invalid_answer: {choice!r} not in options"
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities")
        else:  # noul
            p_yes = answer.get("noul")
            choice = None if p_yes is None else ("yes" if p_yes >= 0.5 else "no")
            if p_yes is None:
                error = "invalid_answer: no 'noul' probability in response"
            confidence = None
            probabilities = (
                {"yes": p_yes, "no": round(1 - p_yes, 4)} if p_yes is not None else None
            )

        return Prediction(
            example_id=example.id,
            model=body.get("model", self.model),
            answer=choice,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=wall_ms,
            eval_ms=(body.get("eval_duration") or 0) / 1e6,
            input_tokens=body.get("usage", {}).get("input_tokens"),
            state_truncated=body.get("state_truncated", False),
            error=error,
            raw=body,
        )
