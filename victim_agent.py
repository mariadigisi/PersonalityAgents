"""
Victim Agent — Phishing Response Framework for MBTI Profiles
============================================================
Simulates recipient decision-making upon receiving phishing vectors.
Each LLM persona evaluates simulated security prompts based on internal cognitive heuristics.

Output Metrics per Trial:
  - decision   : CLICK / NO_CLICK
  - confidence : 1-5 Likert scale rating certainty
  - reasoning  : Natural language decision justification
"""

import re
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Any
from phishing_generator import PhishingEmail, Lever


# ─────────────────────────────────────────────
# 1. RESPONSE DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class VictimResponse:
    agent_name: str
    mbti_type: str
    lever: Lever
    decision: str        # "CLICK" or "NO_CLICK"
    confidence: int      # 1-5
    reasoning: str
    vulnerable: bool = False

    def __post_init__(self) -> None:
        self.vulnerable = (self.decision == "CLICK")


# ─────────────────────────────────────────────
# 2. PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

def build_victim_prompt(email: PhishingEmail) -> str:
    return (
        f"You received the following email in your corporate inbox:\n\n"
        f"FROM: noreply@sistemi-interni.azienda.it\n"
        f"SUBJECT: {email.subject}\n\n"
        f"{email.body}\n\n"
        f"Select your action: 'CLICK' or 'NO_CLICK'.\n"
        f"Respond ONLY in JSON format: "
        f'{{"decision": "CLICK" or "NO_CLICK", "confidence": 1-5, "reasoning": "..."}}'
    )


VICTIM_SYSTEM_PROMPT = """
React to this email authentically and spontaneously, adhering strictly to your persona's profile and behavioral tendencies.

Respond ONLY with valid JSON, providing no supplementary text:
{
  "decision": "CLICK" or "NO_CLICK",
  "confidence": <integer from 1=very uncertain to 5=extremely confident>,
  "reasoning": "<maximum 60 words explaining your internal decision logic>"
}"""


# ─────────────────────────────────────────────
# 3. RESPONSE PARSING & SANITIZATION
# ─────────────────────────────────────────────

def parse_victim_response(raw: str) -> Dict[str, Any]:
    """
    Parses decision, confidence, and reasoning from the agent's output.
    Handles raw JSON, markdown blocks, and regex extraction fallbacks.
    """
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{" in part:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                break

    try:
        data = json.loads(cleaned)
        return {
            "decision":   _normalize_decision(str(data.get("decision", ""))),
            "confidence": _clamp(int(data.get("confidence", 3)), 1, 5),
            "reasoning":  str(data.get("reasoning", "")),
        }
    except Exception:
        pass

    # Regex fallback strategy for unformatted output
    decision = "NO_CLICK"
    if re.search(r'\bCLICK\b', raw, re.IGNORECASE):
        decision = "CLICK"
    if re.search(r'\bNO_CLICK\b', raw, re.IGNORECASE):
        decision = "NO_CLICK"

    conf_match = re.search(r'"confidence"\s*:\s*([1-5])', raw)
    confidence = int(conf_match.group(1)) if conf_match else 3

    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
    reasoning = reason_match.group(1) if reason_match else raw[:150]

    return {"decision": decision, "confidence": confidence, "reasoning": reasoning}


def _normalize_decision(raw: str) -> str:
    raw = raw.upper().strip()
    if "NO" in raw:
        return "NO_CLICK"
    if "CLICK" in raw:
        return "CLICK"
    return "NO_CLICK"


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────
# 4. EXPERIMENTAL TRIAL RUNNERS
# ─────────────────────────────────────────────

def evaluate_victim(
    agent: Any,
    mbti_type: str,
    email: PhishingEmail,
    get_last_speech_fn: Any,
) -> VictimResponse:
    """
    Presents a single phishing vector to a victim agent and logs the behavioral output.
    """
    prompt = build_victim_prompt(email)
    agent.listen_and_act(VICTIM_SYSTEM_PROMPT + "\n\n" + prompt)
    raw = get_last_speech_fn(agent)
    parsed = parse_victim_response(raw)

    return VictimResponse(
        agent_name=agent.name,
        mbti_type=mbti_type,
        lever=email.lever,
        decision=parsed["decision"],
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
    )


def evaluate_all_levers(
    agent: Any,
    mbti_type: str,
    emails: Dict[Lever, PhishingEmail],
    get_last_speech_fn: Any,
) -> List[VictimResponse]:
    """
    Evaluates a target persona across all 7 psychological levers.
    Returns a sequence of VictimResponse record instances.
    """
    responses = []
    for lever, email in emails.items():
        print(f"    🎯 [{mbti_type}] Lever: {lever.value.upper()} ...")
        resp = evaluate_victim(agent, mbti_type, email, get_last_speech_fn)
        icon = "🔴 CLICK" if resp.vulnerable else "🟢 NO_CLICK"
        print(f"       → {icon}  (confidence: {resp.confidence}/5)")
        responses.append(resp)
    return responses