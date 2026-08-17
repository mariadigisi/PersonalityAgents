"""
MBTI Test Runner for TinyTroupe Agents
======================================
Connects TinyTroupe persona instances to the evaluation benchmark pipeline.

Execution Lifecycle per Agent:
  1. Administers 60 Likert items via TinyTroupe (`listen_and_act`).
  2. Extracts numerical rating responses from the agent's textual output.
  3. Computes resulting MBTI dimension scores and personality classification.
  4. Compares obtained results against target persona profiles and prints evaluation reports.
"""

import os
import re
import json
import tiktoken
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# ─────────────────────────────────────────────
# 1. ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────

_orig = tiktoken.encoding_for_model
def _patched(model_name: str):
    try:
        return _orig(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
tiktoken.encoding_for_model = _patched

# ─────────────────────────────────────────────
# 2. IMPORTS & DEPENDENCIES
# ─────────────────────────────────────────────

from tinytroupe.agent import TinyPerson
import tinytroupe

from mbti_evaluator import (
    QUESTIONS,
    MBTI_DESCRIPTIONS,
    Question,
    TestResult,
    parse_answers,
    compute_mbti,
    build_user_prompt,
)

from mbti_tinytroupe_agents import MBTI_PROFILES, create_all_agents


# ─────────────────────────────────────────────
# 3. EVALUATION DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    agent_name: str
    expected_mbti: str
    obtained_mbti: str
    scores: Dict[str, float]
    raw_answers: List[int]
    match: bool = False          # True if all 4 personality poles align
    partial_match: int = 0       # Number of matching poles out of 4 (0-4)

    def __post_init__(self) -> None:
        self.match = (self.expected_mbti == self.obtained_mbti)
        self.partial_match = sum(
            e == o for e, o in zip(self.expected_mbti, self.obtained_mbti)
        )


# ─────────────────────────────────────────────
# 4. AGENT RESPONSE BRIDGE
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """
The agent responded to the following psychological evaluation items.
Convert the agent's responses into a JSON object containing EXACTLY {n} integer ratings from 1 to 5.

Scale:
  1 = Strongly disagree
  2 = Disagree
  3 = Neutral
  4 = Agree
  5 = Strongly agree

If the agent did not clearly answer a item, default to 3 (neutral).

Respond ONLY with valid JSON and no additional text:
{{"answers": [<n1>, <n2>, ..., <n{n}>]}}

Agent response to process:
{agent_response}
"""


def extract_answers_from_text(
    agent: TinyPerson,
    agent_response: str,
    n_questions: int,
) -> List[int]:
    """
    Executes a formatting pass on TinyPerson to parse unstructured output into Likert integers.
    Falls back to regex extraction upon JSON parsing failure.
    """
    extraction_request = EXTRACTION_PROMPT.format(
        n=n_questions,
        agent_response=agent_response,
    )

    agent.listen_and_act(extraction_request)
    last_action = _get_last_speech(agent)

    try:
        return parse_answers(last_action, n_questions)
    except Exception:
        pass

    # Regex fallback to find integer sequences (1-5)
    numbers = [int(m) for m in re.findall(r'\b[1-5]\b', last_action)]
    if len(numbers) >= n_questions:
        return numbers[:n_questions]

    # Neutral padding fallback
    numbers += [3] * (n_questions - len(numbers))
    return numbers[:n_questions]


def _get_last_speech(agent: TinyPerson) -> str:
    """Retrieves the latest TALK action emitted by the TinyPerson agent instance."""
    if hasattr(agent, "_actions_buffer"):
        for action in reversed(agent._actions_buffer):
            if isinstance(action, dict) and action.get("type") == "TALK":
                return action.get("content", "")

    if hasattr(agent, "episodic_memory"):
        buffer = getattr(agent.episodic_memory, "episodic_buffer", [])
        for entry in reversed(buffer):
            if not isinstance(entry, dict):
                continue
            action = entry.get("content", {}).get("action", {})
            if isinstance(action, dict) and action.get("type") == "TALK":
                return action.get("content", "")

    return ""


# ─────────────────────────────────────────────
# 5. SINGLE AGENT TRIAL RUNNER
# ─────────────────────────────────────────────

BATCH_SIZE = 20   # 3 batches of 20 items = 60 total questions


def run_test_on_agent(agent: TinyPerson, expected_mbti: str) -> EvalResult:
    """
    Administers the MBTI evaluation suite to a target TinyTroupe agent.
    Items are batched to maintain optimal context window bounds.
    """
    print(f"\n▶ Evaluating Agent: {agent.name} (Target: {expected_mbti}) ...")

    all_answers: List[int] = []
    batches = [
        QUESTIONS[i : i + BATCH_SIZE]
        for i in range(0, len(QUESTIONS), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        print(f"   📋 Batch {batch_idx + 1}/{len(batches)} ({len(batch)} items) ...")

        lines = [
            f"You are completing a psychological assessment test. "
            f"Respond to each statement with an integer rating from 1 to 5 "
            f"(1=Strongly disagree, 5=Strongly agree). "
            f"Maintain behavioral consistency with your persona.\n"
        ]
        for q in batch:
            lines.append(f"{q.id}. {q.text}")
        lines.append(
            f"\nProvide a list of {len(batch)} comma-separated integers "
            f"or JSON format {{\"answers\": [...]}}."
        )
        prompt = "\n".join(lines)

        agent.listen_and_act(prompt)
        raw_response = _get_last_speech(agent)

        batch_answers = extract_answers_from_text(agent, raw_response, len(batch))
        all_answers.extend(batch_answers)

    scores, mbti = compute_mbti(QUESTIONS, all_answers)

    result = EvalResult(
        agent_name=agent.name,
        expected_mbti=expected_mbti,
        obtained_mbti=mbti,
        scores=scores,
        raw_answers=all_answers,
    )

    _print_single_result(result)
    return result


# ─────────────────────────────────────────────
# 6. REPORTING & OUTPUT FORMATTING
# ─────────────────────────────────────────────

def _print_single_result(r: EvalResult) -> None:
    icon = "✅" if r.match else ("🟡" if r.partial_match >= 3 else "❌")
    desc = MBTI_DESCRIPTIONS.get(r.obtained_mbti, "Unknown")
    print(f"\n{'='*55}")
    print(f"  {icon}  {r.agent_name}")
    print(f"  Expected : {r.expected_mbti}")
    print(f"  Obtained : {r.obtained_mbti}  —  {desc}")
    print(f"  Match    : {r.partial_match}/4 dimensions correct")
    print(f"{'─'*55}")

    labels = {"EI": "E/I", "SN": "S/N", "TF": "T/F", "JP": "J/P"}
    poles  = {"EI": ("E","I"), "SN": ("S","N"), "TF": ("T","F"), "JP": ("J","P")}
    exp    = r.expected_mbti

    dim_map = {"EI": 0, "SN": 1, "TF": 2, "JP": 3}
    for dim, score in r.scores.items():
        dominant = poles[dim][0] if score >= 0 else poles[dim][1]
        expected_pole = exp[dim_map[dim]]
        ok = "✓" if dominant == expected_pole else "✗"
        bar = "█" * min(int(abs(score)), 20)
        print(f"  {ok} {labels[dim]}: {dominant} {bar} ({score:+.1f})")
    print(f"{'='*55}\n")


def print_final_report(results: List[EvalResult]) -> None:
    total   = len(results)
    perfect = sum(1 for r in results if r.match)
    partial = sum(1 for r in results if not r.match and r.partial_match >= 3)
    wrong   = total - perfect - partial
    avg_dim = sum(r.partial_match for r in results) / total if total else 0.0

    print("\n" + "█"*55)
    print("  FINAL BENCHMARK REPORT — MBTI AGENT ACCURACY")
    print("█"*55)
    print(f"  Agents Tested          : {total}")
    print(f"  ✅ Perfect Matches      : {perfect} ({perfect/total*100:.0f}%)")
    print(f"  🟡 Partial Matches (≥3) : {partial} ({partial/total*100:.0f}%)")
    print(f"  ❌ Mismatches          : {wrong}  ({wrong/total*100:.0f}%)")
    print(f"  📊 Average Accuracy    : {avg_dim:.1f} / 4.0 dimensions")
    print("─"*55)
    print(f"  {'Agent':<30} {'Expected':<10} {'Obtained':<10} {'Match'}")
    print("─"*55)
    for r in results:
        icon = "✅" if r.match else ("🟡" if r.partial_match >= 3 else "❌")
        print(f"  {r.agent_name:<30} {r.expected_mbti:<10} {r.obtained_mbti:<10} {icon} {r.partial_match}/4")
    print("█"*55 + "\n")


# ─────────────────────────────────────────────
# 7. EXECUTION PIPELINE
# ─────────────────────────────────────────────

def run_all_tests(mbti_filter: Optional[List[str]] = None) -> List[EvalResult]:
    """
    Runs the evaluation suite across all 16 target agents or a specified subset.

    Args:
        mbti_filter: List of MBTI type strings to filter evaluation (e.g., ["INTJ", "ENFP"]).
    """
    print("🤖 Initializing TinyTroupe agent instances ...\n")
    agents = create_all_agents()

    profiles_to_test = [
        p for p in MBTI_PROFILES
        if mbti_filter is None or p["mbti"] in mbti_filter
    ]

    print(f"\n🧪 Starting evaluation across {len(profiles_to_test)} agent profiles ...")
    results: List[EvalResult] = []

    for profile in profiles_to_test:
        mbti = profile["mbti"]
        agent = agents[mbti]
        result = run_test_on_agent(agent, expected_mbti=mbti)
        results.append(result)

    print_final_report(results)
    return results


if __name__ == "__main__":
    run_all_tests()