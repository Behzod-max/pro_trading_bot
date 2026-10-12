from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass
class AIAnalysisResult:
    enabled: bool
    direction: str
    confidence: float
    decision: str
    summary: str
    risks: list[str]
    raw_text: str = ""


class AIAnalyst:
    """
    AI layer only.
    It does NOT place orders and does NOT override hard risk rules.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value)

    def _fallback(
        self,
        direction: str,
        score: float,
        status: str,
        warnings: list[str],
    ) -> AIAnalysisResult:
        if status in {"CONFIRMED", "HIGH_CONFIDENCE"} and score >= 75:
            decision = "TASDIQLASHGA YAQIN"
        elif status == "EARLY" or score >= 60:
            decision = "KUZATISH"
        else:
            decision = "SAVDO QILMASLIK"

        return AIAnalysisResult(
            enabled=False,
            direction=direction,
            confidence=round(score, 1),
            decision=decision,
            summary=(
                "OpenAI API kaliti mavjud emas. "
                "Qattiq signal/risk qoidalari asosiy qaror sifatida qoldirildi."
            ),
            risks=warnings[:5],
        )

    def analyze(
        self,
        signal_result: Any,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> AIAnalysisResult:
        direction = self._text(
            getattr(signal_result, "direction", "NEUTRAL"),
            "NEUTRAL",
        ).upper()
        score = self._num(
            getattr(signal_result, "score", 0),
            0,
        )
        status = self._text(
            getattr(signal_result, "status", "NO_SIGNAL"),
            "NO_SIGNAL",
        ).upper()

        warnings = list(
            getattr(signal_result, "warnings", []) or []
        )
        reasons = list(
            getattr(signal_result, "reasons", []) or []
        )
        components = getattr(
            signal_result,
            "components",
            {},
        ) or {}

        snapshot = market_snapshot or {}

        if not self.client:
            return self._fallback(
                direction,
                score,
                status,
                warnings,
            )

        payload = {
            "direction": direction,
            "score": score,
            "status": status,
            "reasons": reasons[-20:],
            "warnings": warnings[-20:],
            "components": components,
            "market": snapshot,
        }

        system_prompt = """
You are a conservative professional XAUUSD trading analyst.
Analyze only the supplied data. Never claim certainty or future knowledge.
Do not invent prices, indicators, news, or confirmations.
The hard signal engine and risk manager have priority over your opinion.

Return ONLY valid JSON:
{
  "direction": "BUY|SELL|NEUTRAL",
  "confidence": 0,
  "decision": "BUY|SELL|WAIT|NO_TRADE",
  "summary": "short Uzbek explanation",
  "risks": ["short Uzbek risk", "..."]
}

Rules:
- If the supplied signal status is NO_SIGNAL, prefer WAIT or NO_TRADE.
- If candle confirmation is absent, do not upgrade to a confirmed trade.
- If risk/reward or entry quality is poor, prefer WAIT.
- Confidence must be 0-100.
"""

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                ),
            )

            text = getattr(response, "output_text", "") or ""

            data = json.loads(text)

            ai_direction = str(
                data.get("direction", direction)
            ).upper()

            ai_confidence = max(
                0.0,
                min(
                    100.0,
                    self._num(
                        data.get("confidence"),
                        score,
                    ),
                ),
            )

            decision = str(
                data.get("decision", "WAIT")
            ).upper()

            summary = self._text(
                data.get("summary"),
                "AI tahlil natijasi mavjud.",
            )

            ai_risks = data.get("risks", [])
            if not isinstance(ai_risks, list):
                ai_risks = [str(ai_risks)]

            return AIAnalysisResult(
                enabled=True,
                direction=ai_direction,
                confidence=round(ai_confidence, 1),
                decision=decision,
                summary=summary,
                risks=[str(x) for x in ai_risks[:8]],
                raw_text=text,
            )

        except Exception as exc:
            fallback = self._fallback(
                direction,
                score,
                status,
                warnings,
            )
            fallback.summary = (
                f"AI vaqtincha ishlamadi: {type(exc).__name__}. "
                "Qattiq signal/risk qoidalari o'z kuchida."
            )
            return fallback
