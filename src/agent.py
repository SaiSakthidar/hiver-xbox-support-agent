"""
XboxSupport AI agent with three capabilities:
  1. classify_intent   — maps customer message to one of 8 intents
  2. draft_reply       — RAG-grounded reply using historical Xbox responses
  3. decide_escalation — auto-handle vs escalate, with stated reason

All three functions are context-aware: when prior conversation turns are
available (46% of threads), they are included so the agent can:
  - Classify ambiguous follow-ups correctly ("it still doesn't work")
  - Avoid re-suggesting steps already tried in earlier brand replies
  - Escalate faster when the customer has already gone through troubleshooting

Each function calls the LLM backend configured in .env (anthropic/gemini/ollama).
"""

import ast
import json
import re
from src.llm import chat
from src.retriever import Retriever

INTENTS = [
    "account_access",
    "billing_subscription",
    "hardware_device",
    "network_connectivity",
    "game_content",
    "app_feature",
    "enforcement_ban",
    "other",
]

INTENT_DESCRIPTIONS = {
    "account_access":       "Sign-in failures, password resets, gamertag changes, account settings",
    "billing_subscription": "Refunds, charges, gift cards, Xbox Live Gold, spending limits, purchase errors",
    "hardware_device":      "Controller issues, console won't power on, HDMI/display, physical damage",
    "network_connectivity": "Xbox Live connection, NAT type, error codes, server outages, ping/latency",
    "game_content":         "Game-specific bugs, achievements, clips, backwards compatibility, installs, DLC",
    "app_feature":          "Streaming apps (Netflix, Hulu), Mixer/broadcast, UI complaints, feature requests",
    "enforcement_ban":      "Account bans, suspensions, enforcement team, fraudulent charges, account theft",
    "other":                "Chitchat, thanks, follow-up with no new issue, unclear or off-topic",
}

_INTENT_LIST = "\n".join(f"  - {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())

_retriever: Retriever | None = None

CONTEXT_TURNS = 4   # max prior turns to include (keeps prompts bounded)


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _parse_context(prior_context) -> list[dict]:
    """Parse prior_context field (string repr of list or actual list)."""
    if not prior_context or str(prior_context) in ("[]", "nan", ""):
        return []
    if isinstance(prior_context, list):
        return prior_context
    try:
        return ast.literal_eval(str(prior_context))
    except Exception:
        return []


def _format_context_block(turns: list[dict]) -> str:
    """Format prior turns as a readable conversation block."""
    if not turns:
        return ""
    lines = []
    for t in turns[-CONTEXT_TURNS:]:
        role = "XboxSupport" if not t.get("inbound", True) else "Customer"
        text = str(t.get("text", ""))[:180]
        lines.append(f"  {role}: {text}")
    return "\n".join(lines)


def _extract_tried_steps(turns: list[dict]) -> str:
    """Extract what XboxSupport already suggested in prior turns."""
    suggestions = []
    for t in turns:
        if not t.get("inbound", True):   # brand turn
            text = str(t.get("text", ""))
            if len(text) > 20:
                suggestions.append(f"  - {text[:150]}")
    return "\n".join(suggestions) if suggestions else ""


# ── 1. Intent Classification ──────────────────────────────────────────────────

CLASSIFY_SYSTEM = f"""You are an intent classifier for XboxSupport customer service tweets.

Classify the customer message into EXACTLY ONE of these intents:
{_INTENT_LIST}

DISAMBIGUATION RULES:
- hardware_device vs network_connectivity: if the customer's own console keeps disconnecting, won't complete setup, freezes, or shows hardware symptoms (black screen, disc drive, controller unresponsive), use hardware_device — even if they describe it using wifi/network language. Only use network_connectivity when the issue is clearly a service-wide outage, NAT type, party/chat errors, or Xbox Live error codes affecting multiple users.
- account_access vs network_connectivity: if the customer is locked out of their account or can't sign in, use account_access even if they mention "Xbox Live being down".

EXAMPLES (follow these closely):
Customer: "new Xbox. During setup after entering my wifi password, it connects & then goes back to the wifi screen"
Intent: hardware_device

Customer: "my xbox would disconnect from my wifi, and would not reconnect"
Intent: hardware_device

Customer: "i cannot download anything or stream anything. I've rebooted the console several times"
Intent: hardware_device

Customer: "has Xbox live gone down? I just got threw out of my account and it won't let me back in"
Intent: account_access

Customer: "can't connect to Xbox Live, everyone getting error code 0x87DD0006?"
Intent: network_connectivity

Customer: "NAT type is strict and I can't join my friend's party"
Intent: network_connectivity

When prior conversation context is provided, use it to resolve ambiguous
follow-up messages (e.g. "it still doesn't work" → look at what the prior
issue was to infer the correct intent).

Reply with a JSON object with two fields:
  "intent": one of the exact intent names above
  "confidence": "high" | "medium" | "low"

No other text. JSON only."""


def classify_intent(customer_text: str, prior_context=None) -> dict:
    """Returns {intent, confidence}. Uses prior context when available."""
    turns = _parse_context(prior_context)
    ctx_block = _format_context_block(turns)

    if ctx_block:
        prompt = (
            f"Prior conversation:\n{ctx_block}\n\n"
            f"Latest customer message:\n\"\"\"\n{customer_text}\n\"\"\""
        )
    else:
        prompt = f'Customer message:\n"""\n{customer_text}\n"""'

    raw = chat(prompt, system=CLASSIFY_SYSTEM, max_tokens=64, temperature=0.0)
    try:
        result = json.loads(raw.strip())
        if result.get("intent") not in INTENTS:
            result["intent"] = "other"
        result["had_context"] = bool(ctx_block)
        return result
    except Exception:
        raw_l = (raw or "").lower()
        for intent in INTENTS:
            if intent in raw_l:
                return {"intent": intent, "confidence": "low", "had_context": bool(ctx_block)}
        return {"intent": "other", "confidence": "low", "had_context": bool(ctx_block)}


# ── 2. Reply Drafting (RAG) ────────────────────────────────────────────────────

DRAFT_SYSTEM = """You are drafting customer support replies for XboxSupport on Twitter.

Rules:
- Keep replies under 280 characters when possible; split into parts if needed
- Be concise, empathetic, and actionable
- Ground your reply in the historical examples provided
- Do NOT repeat troubleshooting steps already suggested in this conversation
- Do NOT make up links, phone numbers, or support codes
- Use "@USER" instead of real usernames
- Do not add agent initials or tweet-numbering suffixes"""


def draft_reply(customer_text: str, intent: str, prior_context=None, k: int = 3) -> dict:
    """
    Retrieves k similar historical threads, then drafts a grounded reply.
    Passes prior context so the agent avoids repeating already-tried steps.
    Returns {reply, sources}.
    """
    retriever = _get_retriever()
    hits = retriever.query(customer_text, k=k)

    examples_block = "\n\n".join(
        f"Example {i+1}:\n  Customer: {h['customer_text'][:200]}\n  XboxSupport: {h['brand_reply'][:200]}"
        for i, h in enumerate(hits)
    )

    turns = _parse_context(prior_context)
    ctx_block = _format_context_block(turns)
    tried_block = _extract_tried_steps(turns)

    context_section = ""
    if ctx_block:
        context_section = f"\nConversation so far:\n{ctx_block}\n"
    if tried_block:
        context_section += f"\nSteps already suggested (DO NOT repeat):\n{tried_block}\n"

    prompt = f"""Intent: {intent}
{context_section}
Historical XboxSupport examples for similar issues:
{examples_block}

Now draft a reply for this latest customer message:
\"\"\"{customer_text}\"\"\"

Write ONLY the reply text. No preamble, no explanation."""

    reply = chat(prompt, system=DRAFT_SYSTEM, max_tokens=300, temperature=0.3)
    return {
        "reply": reply.strip(),
        "sources": [h["thread_id"] for h in hits],
    }


# ── 3. Escalation Decision ─────────────────────────────────────────────────────

ESCALATION_SYSTEM = """You are a customer support routing agent for Xbox.

Decide whether this customer message should be:
  - "auto": handled automatically (the drafted reply is sufficient)
  - "escalate": routed to a human agent

ALWAYS escalate if the message involves:
  - Account ban or suspension appeals
  - Fraudulent charges or account theft
  - Hardware damage or warranty claims
  - Billing disputes over a specific amount
  - The customer has already tried multiple troubleshooting steps with no resolution

When prior conversation context is provided, if the customer has been through
2+ rounds of troubleshooting without resolution, lean toward escalation.

Reply with a JSON object:
  "decision": "auto" or "escalate"
  "reason": one short sentence explaining why

JSON only. No other text."""


def decide_escalation(customer_text: str, intent: str, draft_reply_text: str,
                       prior_context=None) -> dict:
    """Returns {decision, reason}. Uses prior context to detect repeat troubleshooting."""
    turns = _parse_context(prior_context)
    ctx_block = _format_context_block(turns)

    context_section = f"\nPrior conversation:\n{ctx_block}\n" if ctx_block else ""

    prompt = f"""Intent: {intent}
{context_section}
Latest customer message: \"\"\"{customer_text}\"\"\"
Drafted reply: \"\"\"{draft_reply_text}\"\"\""""

    raw = chat(prompt, system=ESCALATION_SYSTEM, max_tokens=100, temperature=0.0)
    try:
        result = json.loads(raw.strip())
        if result.get("decision") not in ("auto", "escalate"):
            result["decision"] = "auto"
        return result
    except Exception:
        raw_l = (raw or "").lower()
        decision = "escalate" if "escalate" in raw_l else "auto"
        return {"decision": decision, "reason": (raw or "")[:200]}


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run(customer_text: str, prior_context=None, verbose: bool = False) -> dict:
    """
    End-to-end context-aware pipeline: classify → draft → escalate.

    Args:
        customer_text: the latest customer tweet
        prior_context: list of prior turns (from xbox_threads.csv) or raw string
        verbose: print a summary to stdout

    Returns a structured result dict.
    """
    classification = classify_intent(customer_text, prior_context)
    intent = classification["intent"]

    draft = draft_reply(customer_text, intent, prior_context)
    reply_text = draft["reply"]

    escalation = decide_escalation(customer_text, intent, reply_text, prior_context)

    result = {
        "customer_text": customer_text,
        "had_context": classification.get("had_context", False),
        "intent": intent,
        "confidence": classification["confidence"],
        "reply": reply_text,
        "sources": draft["sources"],
        "decision": escalation["decision"],
        "reason": escalation["reason"],
    }

    if verbose:
        ctx_label = " (with context)" if result["had_context"] else ""
        print(f"\nCustomer : {customer_text}")
        print(f"Intent   : {intent}  ({classification['confidence']}){ctx_label}")
        print(f"Reply    : {reply_text}")
        print(f"Decision : {escalation['decision']} — {escalation['reason']}")

    return result


if __name__ == "__main__":
    import pandas as pd

    # Test with a thread that has real prior context
    df = pd.read_csv("data/xbox_threads.csv")
    has_ctx = df[df["prior_context"].apply(lambda x: str(x) not in ("[]", "nan", ""))].head(3)

    for _, row in has_ctx.iterrows():
        run(row["customer_text"], prior_context=row["prior_context"], verbose=True)
        print()

    # Test without context
    tests = [
        "my xbox won't turn on at all, just a black screen",
        "I got charged twice for Xbox Live Gold, need a refund",
    ]
    for t in tests:
        run(t, verbose=True)
        print()
