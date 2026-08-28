"""
Expert Agent — paper Section II-B-6 (Eq. 19).

Non-intervening by design: called ONLY for human-facing explanation when
warning_level is LOW or MEDIUM (per Algorithm 2 / Section IV-D). Never touches
final_warning / final_failure / alert_level.

Paper uses the OpenAI Responses API with gpt-4o-mini, JSON-only schema,
max 400 output tokens, 2 retries, two-stage parse fallback (direct JSON, then
bracket extraction). Reproduced below with the `openai` Python SDK.
"""
import json
import os
from dataclasses import dataclass

SCHEMA_FIELDS = ["summary", "exp", "likely_fault", "recommended_action", "severity"]
ALLOWED_ACTIONS = {"MONITOR", "ACK_AND_INVESTIGATE", "SCHEDULE_MAINTENANCE", "IMMEDIATE_SHUTDOWN"}
ALLOWED_SEVERITY = {"LOW", "MEDIUM", "HIGH"}

SYSTEM_PROMPT = """You are the Expert Agent inside ASPIRE, an industrial IIoT \
predictive-maintenance decision system. You reason over DECISION EVIDENCE, not \
raw sensor data. You never change or influence any automated decision. \
Respond with STRICT JSON ONLY, no prose, no markdown fences, matching exactly:
{"summary": str, "exp": str, "likely_fault": str, "recommended_action": \
"MONITOR"|"ACK_AND_INVESTIGATE"|"SCHEDULE_MAINTENANCE"|"IMMEDIATE_SHUTDOWN", \
"severity": "LOW"|"MEDIUM"|"HIGH"}
If no plausible physical fault mechanism is evident, set likely_fault to "Unknown"."""


def _build_user_prompt(decision_packet: dict) -> str:
    return (
        "Decision context (do not modify, only explain):\n"
        + json.dumps(decision_packet, default=str, indent=2)
    )


def _parse_response(text: str) -> dict:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Could not parse Expert Agent response: {text[:200]}")
        obj = json.loads(text[start:end + 1])
    for field_ in SCHEMA_FIELDS:
        obj.setdefault(field_, "Unknown" if field_ == "likely_fault" else "")
    if obj.get("recommended_action") not in ALLOWED_ACTIONS:
        obj["recommended_action"] = "MONITOR"
    if obj.get("severity") not in ALLOWED_SEVERITY:
        obj["severity"] = "LOW"
    return obj


@dataclass
class ExpertAgent:
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 400
    max_retries: int = 2

    def invoke(self, decision_packet: dict) -> dict:
        """
        decision_packet should include: final_warning/failure/anomaly/drift,
        alert_level, [P_normal, P_warn, P_fault], detection/drift risk, FDS/FDI/
        WSS/WDI, a short sensor snapshot, and recent decision history — per Sec II-B-6.
        Returns a fallback structured object on total failure (never raises to caller).
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception as e:
            return self._fallback(f"OpenAI client unavailable: {e}")

        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                resp = client.responses.create(
                    model=self.model,
                    max_output_tokens=self.max_output_tokens,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(decision_packet)},
                    ],
                )
                text = resp.output_text
                return _parse_response(text)
            except Exception as e:  # noqa: BLE001 — graceful fallback per paper design
                last_err = e
                continue
        return self._fallback(f"Expert Agent failed after retries: {last_err}")

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "summary": "Expert Agent unavailable — automated decision stands unaffected.",
            "exp": reason,
            "likely_fault": "Unknown",
            "recommended_action": "MONITOR",
            "severity": "LOW",
        }
