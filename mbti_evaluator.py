"""
MBTI Personality Evaluation Benchmark for LLM Agents
=====================================================
A benchmark framework for assessing 16-Personalities Myers-Briggs Type Indicator 
(MBTI) dimensions in Large Language Model profiles.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, List, Tuple


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

Dimension = Literal["EI", "SN", "TF", "JP"]
Pole      = Literal[1, -1]   # +1 = primary pole (E,S,T,J), -1 = secondary (I,N,F,P)

@dataclass
class Question:
    id: int
    text: str
    dimension: Dimension
    pole: Pole            # +1 if "agree" points towards E/S/T/J, -1 towards I/N/F/P

@dataclass
class AgentProfile:
    """Represents the persona profile prompt fed to the LLM agent."""
    name: str
    description: str      # e.g., "You are an analytical and reserved scientist."

@dataclass
class TestResult:
    agent_name: str
    scores: Dict[Dimension, float] = field(default_factory=dict)
    mbti_type: str = ""
    raw_answers: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────
# 2. QUESTION BANK (60 items, 15 per dimension)
# ─────────────────────────────────────────────

QUESTIONS: List[Question] = [
    # ── Extraversion / Introversion (E/I) ─────
    #15 questions ispired by MBTI official test

    # ── Sensing / Intuition (S/N) ──────────────
    #15 questions ispired by MBTI official test

    # ── Thinking / Feeling (T/F) ───────────────
    #15 questions ispired by MBTI official test
    
    # ── Judging / Perceiving (J/P) ────────────
    #15 questions ispired by MBTI official test
]


# ─────────────────────────────────────────────
# 3. LLM API CLIENT WRAPPER
# ─────────────────────────────────────────────

class LLMClient:
    """Wrapper to handle inference across different API providers."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider.lower()
        self.model = model

    def query(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# ─────────────────────────────────────────────
# 4. EVALUATION & SCORING LOGIC
# ─────────────────────────────────────────────

def build_system_prompt(profile: AgentProfile) -> str:
    return f"""You are an agent with the following personality profile: {profile.description}

You are completing a psychological personality assessment.
Answer authentically and consistently with your assigned persona.
Respond ONLY with a valid JSON object, without markdown blocks or supplementary text, using the following schema:
{{"answers": [<number>, <number>, ...]}}
Where each entry is an integer from 1 to 5 corresponding to:
  1 = Strongly disagree
  2 = Disagree
  3 = Neutral
  4 = Agree
  5 = Strongly agree"""


def build_user_prompt(questions: List[Question]) -> str:
    lines = ["Respond to each of the following statements with an integer rating from 1 to 5:\n"]
    for q in questions:
        lines.append(f"{q.id}. {q.text}")
    lines.append(f'\nProvide exactly {len(questions)} ratings in the "answers" key array.')
    return "\n".join(lines)


def parse_answers(response: str, expected_count: int) -> List[int]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    answers = data["answers"]

    if len(answers) != expected_count:
        raise ValueError(f"Expected {expected_count} ratings, got {len(answers)}")

    for i, a in enumerate(answers):
        if not (1 <= int(a) <= 5):
            raise ValueError(f"Answer at index {i+1} out of bounds (1-5): {a}")

    return [int(a) for a in answers]


def compute_mbti(questions: List[Question], answers: List[int]) -> Tuple[Dict[Dimension, float], str]:
    scores: Dict[Dimension, float] = {"EI": 0.0, "SN": 0.0, "TF": 0.0, "JP": 0.0}

    for q, ans in zip(questions, answers):
        contribution = (ans - 3) * q.pole
        scores[q.dimension] += contribution

    mbti = ""
    mbti += "E" if scores["EI"] >= 0 else "I"
    mbti += "S" if scores["SN"] >= 0 else "N"
    mbti += "T" if scores["TF"] >= 0 else "F"
    mbti += "J" if scores["JP"] >= 0 else "P"

    return scores, mbti


MBTI_DESCRIPTIONS = {
    "INTJ": "The Architect — strategic, independent, determined.",
    "INTP": "The Logician — analytical, curious, lover of abstract ideas.",
    "ENTJ": "The Commander — decisive, natural leader, goal-oriented.",
    "ENTP": "The Debater — creative, witty, loves challenging conventions.",
    "INFJ": "The Advocate — visionary, empathetic, value-driven.",
    "INFP": "The Mediator — idealistic, creative, deeply empathetic.",
    "ENFJ": "The Protagonist — charismatic, altruistic, inspirational.",
    "ENFP": "The Campaigner — enthusiastic, creative, loves human connections.",
    "ISTJ": "The Logistician — reliable, methodical, respects traditions.",
    "ISFJ": "The Defender — caring, loyal, attentive to others' needs.",
    "ESTJ": "The Executive — organized, direct, respects rules and structures.",
    "ESFJ": "The Consul — sociable, caring, community-oriented.",
    "ISTP": "The Virtuoso — practical, observant, skilled with tools.",
    "ISFP": "The Adventurer — flexible, artistic, lives in the moment.",
    "ESTP": "The Entrepreneur — energetic, pragmatic, loves action.",
    "ESFP": "The Entertainer — spontaneous, enthusiastic, loves social life.",
}


def print_result(result: TestResult) -> None:
    desc = MBTI_DESCRIPTIONS.get(result.mbti_type, "Unrecognized type")
    print(f"\n{'='*60}")
    print(f"  Agent Name : {result.agent_name}")
    print(f"  MBTI Type  : {result.mbti_type} — {desc}")
    print(f"{'─'*60}")
    print("  Dimension Scores (>0 indicates primary pole):")
    labels = {"EI": "E/I", "SN": "S/N", "TF": "T/F", "JP": "J/P"}
    poles  = {"EI": ("E","I"), "SN": ("S","N"), "TF": ("T","F"), "JP": ("J","P")}
    for dim, score in result.scores.items():
        dominant = poles[dim][0] if score >= 0 else poles[dim][1]
        bar = "█" * min(int(abs(score)), 20)
        print(f"  {labels[dim]}: {dominant} {bar} ({score:+.1f})")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 5. EXECUTION PIPELINE
# ─────────────────────────────────────────────

def run_test(profile: AgentProfile, client: LLMClient) -> TestResult:
    print(f"\n▶ Evaluating Profile: {profile.name} ...")

    system_prompt = build_system_prompt(profile)
    user_prompt   = build_user_prompt(QUESTIONS)

    raw_response = client.query(system_prompt, user_prompt)
    answers      = parse_answers(raw_response, len(QUESTIONS))
    scores, mbti = compute_mbti(QUESTIONS, answers)

    result = TestResult(
        agent_name=profile.name,
        scores=scores,
        mbti_type=mbti,
        raw_answers=answers,
    )
    print_result(result)
    return result


def run_benchmark(profiles: List[AgentProfile], client: LLMClient) -> List[TestResult]:
    results = [run_test(p, client) for p in profiles]

    print("\n" + "="*60)
    print("  BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r.agent_name:<35} → {r.mbti_type}")
    print("="*60 + "\n")

    return results
