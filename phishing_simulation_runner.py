"""
MBTI Phishing Simulation — Runner & Orchestration Pipeline
==========================================================
Orchestrates the social engineering simulation experiment and generates execution reports.

Execution Lifecycle:
  1. Attacker agent generates 7 phishing email vectors (one per psychological lever).
  2. Synthetic victim agents evaluate all 7 email vectors (Output: CLICK / NO_CLICK).
  3. Constructs a structured persona-by-lever vulnerability matrix.
  4. Renders output metrics to stdout and exports dataset records to CSV and JSON formats.

Output Artifacts:
  - phishing_report.csv  — Raw vulnerability matrix (for spreadsheet analysis)
  - phishing_report.json — Complete dataset including natural language decision justifications
"""

import os
import re
import csv
import json
import tiktoken
from collections import defaultdict
from typing import List, Dict, Optional, Any

# ─────────────────────────────────────────────
# 1. ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────

# Monkey-patch tiktoken encoding resolution for Gemini model compatibility
_orig = tiktoken.encoding_for_model
def _patched(model_name: str):
    try:
        return _orig(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
tiktoken.encoding_for_model = _patched

# ─────────────────────────────────────────────
# 2. DEPENDENCIES & INITIALIZATION
# ─────────────────────────────────────────────

from tinytroupe.agent import TinyPerson
import tinytroupe
tinytroupe.config.set("OpenAI", "MODEL", "gemini-2.5-flash")

from phishing_generator import (
    Lever, PhishingEmail,
    create_attacker_agent, generate_all_emails,
)
from victim_agent import (
    VictimResponse, evaluate_all_levers, evaluate_victim
)
from mbti_tinytroupe_agents import MBTI_PROFILES, create_all_agents


# ─────────────────────────────────────────────
# 3. UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_last_speech(agent: TinyPerson) -> str:
    """Retrieves the latest TALK action content from the agent's action buffer."""
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


def parse_json_response(raw: str) -> Dict[str, Any]:
    """Cleans and parses JSON responses from agent text outputs."""
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
        return json.loads(cleaned)
    except Exception:
        return {}


# ─────────────────────────────────────────────
# 4. REPORTING & ANALYTICS
# ─────────────────────────────────────────────

LEVER_ORDER = list(Lever)
MBTI_GROUPS = {
    "Analysts":   ["INTJ", "INTP", "ENTJ", "ENTP"],
    "Diplomats":  ["INFJ", "INFP", "ENFJ", "ENFP"],
    "Sentinels":  ["ISTJ", "ISFJ", "ESTJ", "ESFJ"],
    "Explorers":  ["ISTP", "ISFP", "ESTP", "ESFP"],
}


def build_matrix(
    all_responses: List[VictimResponse],
) -> Dict[str, Dict[Lever, VictimResponse]]:
    """Constructs a nested mapping: MBTI_type -> Lever -> VictimResponse."""
    matrix: Dict[str, Dict[Lever, VictimResponse]] = defaultdict(dict)
    for r in all_responses:
        matrix[r.mbti_type][r.lever] = r
    return matrix


def print_console_report(
    matrix: Dict[str, Dict[Lever, VictimResponse]],
    emails: Dict[Lever, PhishingEmail],
) -> None:
    """Renders the vulnerability report directly to stdout."""

    lever_labels = {l: l.value[:7].upper() for l in LEVER_ORDER}

    header = f"  {'Profile':<14}" + "".join(f"  {lever_labels[l]:<7}" for l in LEVER_ORDER) + "  TOTAL"
    sep    = "─" * len(header)

    print("\n" + "█"*70)
    print("  SIMULATION REPORT: PHISHING VULNERABILITY BY MBTI PROFILE")
    print("█"*70)
    print(f"\n  Legend: 🔴 CLICK (Vulnerable)  |  🟢 NO_CLICK (Resistant)\n")
    print(f"  {'Levers:':<16}" + "".join(f"  {lever_labels[l]:<7}" for l in LEVER_ORDER))
    print(sep)

    overall_vuln: Dict[Lever, int] = defaultdict(int)
    overall_total = 0

    for group_name, mbti_list in MBTI_GROUPS.items():
        print(f"\n  ── {group_name} ──")
        for mbti in mbti_list:
            if mbti not in matrix:
                continue
            row_data = matrix[mbti]
            cells = ""
            vuln_count = 0
            for lever in LEVER_ORDER:
                resp = row_data.get(lever)
                if resp:
                    icon = "🔴" if resp.vulnerable else "🟢"
                    if resp.vulnerable:
                        vuln_count += 1
                        overall_vuln[lever] += 1
                    cells += f"  {icon}     "
                else:
                    cells += "  ??     "
            overall_total += 1
            pct = int(vuln_count / len(LEVER_ORDER) * 100)
            bar = "▓" * (pct // 10)
            print(f"  {mbti:<14}{cells}  {vuln_count}/{len(LEVER_ORDER)} ({pct:3d}%) {bar}")

    print(f"\n{sep}")
    print(f"  {'VULN/TOTAL':<14}" + "".join(
        f"  {overall_vuln[l]}/{overall_total}  " for l in LEVER_ORDER
    ))
    pcts = {l: int(overall_vuln[l] / overall_total * 100) if overall_total else 0
            for l in LEVER_ORDER}
    print(f"  {'%':<14}" + "".join(f"  {pcts[l]:3d}%   " for l in LEVER_ORDER))
    print("█"*70)

    sorted_levers = sorted(LEVER_ORDER, key=lambda l: overall_vuln[l], reverse=True)
    print("\n  📊 MOST EFFECTIVE PSYCHOLOGICAL LEVERS:")
    for i, lever in enumerate(sorted_levers, 1):
        pct = pcts[lever]
        bar = "█" * (pct // 5)
        print(f"  {i}. {lever.value.upper():<12} {bar} {pct}%")

    print("\n  🎯 MOST VULNERABLE PERSONA PROFILES:")
    vuln_by_profile = {}
    for mbti, row_data in matrix.items():
        clicks = sum(1 for r in row_data.values() if r.vulnerable)
        vuln_by_profile[mbti] = clicks
    for mbti, clicks in sorted(vuln_by_profile.items(), key=lambda x: -x[1]):
        pct = int(clicks / len(LEVER_ORDER) * 100)
        bar = "█" * (pct // 10)
        print(f"  {mbti:<6} {bar} {pct}% ({clicks}/{len(LEVER_ORDER)} levers)")

    print()


def print_reasoning_detail(
    matrix: Dict[str, Dict[Lever, VictimResponse]],
    mbti_filter: Optional[List[str]] = None,
) -> None:
    """Prints qualitative reasoning logs for each agent persona and psychological lever."""
    print("\n" + "="*70)
    print("  QUALITATIVE REASONING LOGS")
    print("="*70)
    profiles = mbti_filter or list(matrix.keys())
    for mbti in profiles:
        if mbti not in matrix:
            continue
        print(f"\n  [{mbti}]")
        for lever in LEVER_ORDER:
            resp = matrix[mbti].get(lever)
            if not resp:
                continue
            icon = "🔴" if resp.vulnerable else "🟢"
            print(f"  {icon} {lever.value.upper():<12} (confidence: {resp.confidence}/5)")
            print(f"     \"{resp.reasoning}\"")


def save_csv(
    matrix: Dict[str, Dict[Lever, VictimResponse]], 
    path: str = "phishing_report.csv"
) -> None:
    """Exports the vulnerability matrix to a CSV file."""
    lever_names = [l.value for l in LEVER_ORDER]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["MBTI"] + lever_names + ["VULN_TOTAL", "VULN_PCT"])
        for mbti, row_data in matrix.items():
            row = [mbti]
            vuln = 0
            for lever in LEVER_ORDER:
                resp = row_data.get(lever)
                if resp:
                    row.append(1 if resp.vulnerable else 0)
                    if resp.vulnerable:
                        vuln += 1
                else:
                    row.append("")
            row += [vuln, f"{int(vuln/len(LEVER_ORDER)*100)}%"]
            writer.writerow(row)
    print(f"  💾 Saved CSV report: {path}")


def save_json(
    all_responses: List[VictimResponse],
    emails: Dict[Lever, PhishingEmail],
    path: str = "phishing_report.json",
) -> None:
    """Exports raw experimental outputs and emails to a structured JSON file."""
    data = {
        "emails_generated": {
            lever.value: {
                "subject": email.subject,
                "body": email.body,
            }
            for lever, email in emails.items()
        },
        "responses": [
            {
                "agent":      r.agent_name,
                "mbti":       r.mbti_type,
                "lever":      r.lever.value,
                "decision":   r.decision,
                "vulnerable": r.vulnerable,
                "confidence": r.confidence,
                "reasoning":  r.reasoning,
            }
            for r in all_responses
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved JSON report: {path}")


# ─────────────────────────────────────────────
# 5. EXPERIMENT EXECUTION PIPELINE
# ─────────────────────────────────────────────

def run_phishing_simulation(
    mbti_filter: Optional[List[str]] = None,
    print_reasoning: bool = False,
    save_files: bool = True,
) -> List[VictimResponse]:
    """
    Executes the end-to-end phishing benchmark simulation.
    Ensures isolated state initialization for every individual trial run.
    """

    # ── Phase 1: Attack Generation ──────────────────────────────────────────
    print("\n" + "="*70)
    print("🎭 PHASE 1 — Attacker Agent Generating Phishing Email Vectors ...")
    print("="*70)
    emails = generate_all_emails(None, get_last_speech, parse_json_response)

    print("\n📬 Generated Phishing Vectors:")
    for lever, email in emails.items():
        print(f"\n{email}")

    # ── Phase 2: Victim Persona Selection ──────────────────────────────────
    print("\n" + "="*70)
    print("🤖 PHASE 2 — Initializing Victim Personas ...")
    print("="*70)

    profiles_to_test = [
        p for p in MBTI_PROFILES
        if mbti_filter is None or p["mbti"] in mbti_filter
    ]
    print(f"  Active victim profiles configured for trial: {len(profiles_to_test)}")

    # ── Phase 3: Trial Execution (Strict Memory Isolation) ──────────────────
    print("\n" + "="*70)
    print("📨 PHASE 3 — Administering Vectors to Victim Personas ...")
    print("="*70)

    all_responses: List[VictimResponse] = []
    _agent_counter = 0

    for profile in profiles_to_test:
        mbti = profile["mbti"]
        print(f"\n  🧑 Testing Profile: {profile['name']} [{mbti}]")

        for lever, email in emails.items():
            _agent_counter += 1
            
            # Isolated agent instantiation per trial to prevent memory accumulation
            agent_name = f"{profile['name'].replace(' ', '_')}_{lever.value}_{_agent_counter}"
            agent = TinyPerson(agent_name)
            agent.define("age", profile["age"])
            agent.define("nationality", "Italian")
            agent.define("occupation", profile["occupation"])
            agent.define("personality_traits", [{"trait": t} for t in profile["traits"]])

            print(f"    🎯 [{mbti}] Administering Lever: {lever.value.upper()} ...")
            
            resp = evaluate_victim(
                agent=agent, 
                mbti_type=mbti, 
                email=email, 
                get_last_speech_fn=get_last_speech
            )
            
            icon = "🔴 CLICK" if resp.vulnerable else "🟢 NO_CLICK"
            print(f"       → {icon}  (confidence: {resp.confidence}/5)")
            all_responses.append(resp)

    # ── Phase 4: Output Processing & Reporting ──────────────────────────────
    matrix = build_matrix(all_responses)
    print_console_report(matrix, emails)

    if print_reasoning:
        print_reasoning_detail(matrix, mbti_filter)

    if save_files:
        print("\n  💾 Exporting execution artifacts ...")
        save_csv(matrix)
        save_json(all_responses, emails)

    return all_responses


if __name__ == "__main__":
    run_phishing_simulation()