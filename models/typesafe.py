"""Model adapter for TypeSafe-compatible decision servers.

Speaks POST /v1/systemone, as served by:
  - kev.serve (github.com/jaredpalmer/kev) — Apple Silicon via MLX
  - the decider mlx-vlm shim (github.com/... local serve script)

Same conceptual operation as every other adapter:
    decide(example) -> prediction
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from core.schemas import DecisionExample, Prediction


@dataclass
class TypeSafeConfig:
    host: str = "http://localhost:8009"
    request_model: str = "kev-latest"  # kev.serve serves its checkpoint under this name
    timeout_s: float = 180.0
    api_key: str | None = None  # Bearer auth for the hosted TypeSafe API; never persisted

    def headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        return {"Authorization": f"Bearer {key}"} if key else {}


class TypeSafeServeModel:
    def __init__(self, model: str, config: TypeSafeConfig | None = None):
        self.model = model  # benchmark spec, recorded in predictions
        self.config = config or TypeSafeConfig()
        self._client = httpx.Client(
            base_url=self.config.host,
            timeout=self.config.timeout_s,
            headers=self.config.headers(),
        )

    def models(self) -> dict:
        return self._client.get("/v1/models").json()

    def warmup(self) -> None:
        """Some servers load lazily; a models call is a cheap nudge. Ignore errors."""
        try:
            self.models()
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
        else:  # noul
            options = ["yes", "no"]
            question_spec = {
                "type": "noul",
                "instructions": decision.get("instructions", "Does the statement hold?"),
            }

        payload = {
            "state": example.input["text"],
            "model": self.config.request_model,
            "questions": {question: question_spec},
        }

        t0 = time.perf_counter()
        try:
            resp = self._client.post("/v1/systemone", json=payload)
            wall_ms = (time.perf_counter() - t0) * 1000
            body = resp.json()
        except Exception as exc:
            return Prediction(
                example_id=example.id, model=self.model, answer=None,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"request_failed: {exc}",
            )

        if resp.status_code != 200:
            return Prediction(
                example_id=example.id, model=self.model, answer=None,
                latency_ms=wall_ms,
                error=f"http_{resp.status_code}: {body.get('error', body) if isinstance(body, dict) else body}",
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
            model=self.model,
            answer=choice,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=wall_ms,
            eval_ms=body.get("latency_ms"),  # server-reported model time
            input_tokens=body.get("usage", {}).get("input_tokens"),
            state_truncated=False,
            error=error,
            raw=body,
        )
