"""
Attacker Agent — Simulated Phishing Email Generator
====================================================
Framework for security awareness research and empirical evaluation.
Generated emails are fictitious, generic, and free of real-world brand cloning.

The attacker model operates without prior knowledge of the target's personality (MBTI).
Phishing vectors are generated purely based on assigned psychological levers.

Psychological Levers (Cialdini + Phishing Literature):
  1. URGENCY     — Imminent threat, loss aversion, time pressure.
  2. AUTHORITY   — Perceived authority (IT, executive, compliance).
  3. REWARD      — Gain, compensation, financial incentive.
  4. CURIOSITY   — Confidentiality, exclusive or mysterious insights.
  5. RECIPROCITY — Debt, obligation, favors pre-rendered.
  6. SCARCITY    — Limited availability, exclusive access.
  7. SOCIAL      — Social proof, peer pressure, group compliance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Any


# ─────────────────────────────────────────────
# 1. PSYCHOLOGICAL LEVERS
# ─────────────────────────────────────────────

class Lever(str, Enum):
    URGENCY     = "urgency"
    AUTHORITY   = "authority"
    REWARD      = "reward"
    CURIOSITY   = "curiosity"
    RECIPROCITY = "reciprocity"
    SCARCITY    = "scarcity"
    SOCIAL      = "social"


LEVER_DESCRIPTIONS: Dict[Lever, str] = {
    Lever.URGENCY: (
        "Urgency and loss aversion. Creates an imminent threat: "
        "an account compromise, an unyielding deadline, or an immediate loss. "
        "Goal: Trigger impulsive action without reflection."
    ),
    Lever.AUTHORITY: (
        "Authority and legitimacy. Simulates a communication from an authoritative role: "
        "IT lead, executive manager, compliance officer, or security unit. "
        "Goal: Enforce immediate compliance."
    ),
    Lever.REWARD: (
        "Reward and financial incentive. Promises a tangible benefit: a refund, "
        "a gift card, a bonus payout, or an exclusive offer. "
        "Goal: Leverage gain-seeking motivation."
    ),
    Lever.CURIOSITY: (
        "Curiosity and confidential information. Hints at mysterious or restricted content: "
        "who viewed a profile, a shared private document, or confidential news. "
        "Goal: Exploit the need to know."
    ),
    Lever.RECIPROCITY: (
        "Reciprocity and obligation. Frames an action as already performed on "
        "behalf of the recipient (a sent gift, an activated service, a favor). "
        "Goal: Establish a sense of obligation to reciprocate."
    ),
    Lever.SCARCITY: (
        "Scarcity and exclusivity. Highlights that the opportunity is limited, "
        "reserved exclusively for the recipient, or about to expire. "
        "Goal: Trigger fear of missing out (FOMO)."
    ),
    Lever.SOCIAL: (
        "Social proof and compliance. Mentions that peers (colleagues, team members, "
        "or group members) have already completed the required action. "
        "Goal: Capitalize on conformity and belonging needs."
    ),
}


# ─────────────────────────────────────────────
# 2. EMAIL DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PhishingEmail:
    lever: Lever
    subject: str
    body: str

    def __str__(self) -> str:
        return (
            f"[LEVER: {self.lever.value.upper()}]\n"
            f"SUBJECT: {self.subject}\n"
            f"{'─'*50}\n"
            f"{self.body}\n"
            f"{'─'*50}"
        )


# ─────────────────────────────────────────────
# 3. GENERATION PROMPTS
# ─────────────────────────────────────────────

ATTACKER_SYSTEM_PROMPT = """You are a cybersecurity researcher generating simulated phishing emails for research purposes.

Strict Rules:
- All emails must be fictitious and intended solely for research/educational evaluations.
- DO NOT use real brand names (e.g., no PayPal, no Microsoft, no actual financial institutions).
- Use generic internal senders such as "Internal Technical Support", "HR Team", "IT Security".
- DO NOT include active URLs or functioning links; use placeholders like [LINK].
- Maintain a plausible and professional tone.
- You have zero prior knowledge regarding the recipient's identity or personality.

Respond ONLY with a valid JSON object using the following structure:
{
  "subject": "<email subject line>",
  "body": "<email body text, max 150 words>"
}"""


def build_attacker_prompt(lever: Lever) -> str:
    return (
        f"Generate a simulated phishing email exploiting the following psychological lever:\n\n"
        f"LEVER: {lever.value.upper()}\n"
        f"DESCRIPTION: {LEVER_DESCRIPTIONS[lever]}\n\n"
        f"The email must appear to originate from an internal corporate entity within a generic enterprise. "
        f"Remember: No real brand names, no live URLs, use [LINK] placeholders for hyperlinks."
    )


# ─────────────────────────────────────────────
# 4. GENERATOR (TinyTroupe Bridge Integration)
# ─────────────────────────────────────────────

def create_attacker_agent():
    """Instantiates a TinyTroupe attacker agent instance."""
    from tinytroupe.agent import TinyPerson
    attacker = TinyPerson("PhishingResearcher")
    attacker.define("occupation", "Cybersecurity Researcher")
    attacker.define("personality_traits", [
        {"trait": "You are an expert in social engineering and the psychology of persuasion."},
    ])
    return attacker


_attacker_counter = 0

def _fresh_attacker():
    """
    Instantiates an attacker agent with a unique identifier.
    TinyTroupe enforces a global registry where duplicate names raise a ValueError.
    """
    global _attacker_counter
    _attacker_counter += 1
    from tinytroupe.agent import TinyPerson
    attacker = TinyPerson(f"PhishingResearcher_{_attacker_counter}")
    attacker.define("occupation", "Cybersecurity Researcher")
    attacker.define("personality_traits", [
        {"trait": "You are an expert in social engineering and persuasion psychology."},
    ])
    return attacker


def generate_phishing_email(
    lever: Lever,
    get_last_speech_fn: Callable[[Any], str],
    parse_json_fn: Callable[[str], Dict[str, Any]],
) -> PhishingEmail:
    """
    Generates a simulated phishing email for a targeted psychological lever.
    Instantiates a isolated agent instance per lever to ensure clean context buffers.
    """
    attacker = _fresh_attacker()
    full_prompt = (
        ATTACKER_SYSTEM_PROMPT
        + "\n\n"
        + build_attacker_prompt(lever)
        + "\n\nRespond ONLY with the requested JSON structure. Do not add supplementary text."
    )
    attacker.listen_and_act(full_prompt)
    raw = get_last_speech_fn(attacker)
    data = parse_json_fn(raw)

    # Fallback retry mechanism if JSON output is empty or malformed
    if not data.get("subject"):
        retry_prompt = (
            f'Generate ONLY this JSON structure for a simulated email using lever {lever.value.upper()}:\n'
            f'{{"subject": "<subject>", "body": "<body max 100 words>"}}\n'
            f'Provide raw JSON only.'
        )
        attacker2 = _fresh_attacker()
        attacker2.listen_and_act(retry_prompt)
        raw2 = get_last_speech_fn(attacker2)
        data = parse_json_fn(raw2)

    return PhishingEmail(
        lever=lever,
        subject=data.get("subject", f"[{lever.value.upper()}] Action Required"),
        body=data.get("body", f"Simulated email body for lever {lever.value}. [LINK]"),
    )


def generate_all_emails(
    attacker: Any,           # Maintained for caller interface compatibility
    get_last_speech_fn: Callable[[Any], str],
    parse_json_fn: Callable[[str], Dict[str, Any]],
) -> Dict[Lever, PhishingEmail]:
    """Generates simulated attack vectors for all 7 psychological levers."""
    emails = {}
    for lever in Lever:
        print(f"  📧 Generating email vector [{lever.value.upper()}] ...")
        emails[lever] = generate_phishing_email(
            lever, get_last_speech_fn, parse_json_fn
        )
    return emails