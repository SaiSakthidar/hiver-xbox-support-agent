"""
Interactive demo — type any customer message and see the full agent pipeline.

Usage:
    python3 -m src.demo

Type 'quit' or Ctrl+C to exit.
Type 'context' to enter a multi-turn conversation (simulates prior turns).
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))


import sys
from src.agent import run, INTENTS, INTENT_DESCRIPTIONS

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         XboxSupport AI Agent — Interactive Demo          ║
║                                                          ║
║  Commands:                                               ║
║    Type any customer message and press Enter             ║
║    'intents'  — show intent taxonomy                     ║
║    'quit'     — exit                                     ║
╚══════════════════════════════════════════════════════════╝
"""

INTENT_COLOR = {
    "account_access":       "\033[94m",   # blue
    "billing_subscription": "\033[93m",   # yellow
    "hardware_device":      "\033[91m",   # red
    "network_connectivity": "\033[96m",   # cyan
    "game_content":         "\033[92m",   # green
    "app_feature":          "\033[95m",   # magenta
    "enforcement_ban":      "\033[31m",   # dark red
    "other":                "\033[90m",   # grey
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _color_intent(intent: str) -> str:
    c = INTENT_COLOR.get(intent, "")
    return f"{c}{BOLD}{intent}{RESET}"


def _color_decision(decision: str) -> str:
    if decision == "escalate":
        return f"\033[91m{BOLD}ESCALATE{RESET}"
    return f"\033[92m{BOLD}AUTO-HANDLE{RESET}"


def show_result(result: dict) -> None:
    print()
    print(f"  Intent    : {_color_intent(result['intent'])}  "
          f"{DIM}({result['confidence']} confidence){RESET}"
          + (f"  {DIM}[used context]{RESET}" if result.get("had_context") else ""))
    print(f"  Decision  : {_color_decision(result['decision'])}")
    print(f"  Reason    : {DIM}{result['reason']}{RESET}")
    print()
    print(f"  {BOLD}Draft reply:{RESET}")
    # Wrap reply at 70 chars
    reply = result["reply"]
    for i in range(0, len(reply), 70):
        print(f"    {reply[i:i+70]}")
    print()


def show_intents() -> None:
    print()
    for intent, desc in INTENT_DESCRIPTIONS.items():
        print(f"  {_color_intent(intent):<30}  {DIM}{desc}{RESET}")
    print()


def main():
    print(BANNER)

    prior_turns = []

    while True:
        try:
            prompt = input(f"{BOLD}Customer>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if prompt.lower() == "intents":
            show_intents()
            continue

        print(f"  {DIM}Thinking…{RESET}", end="\r")

        prior_context = prior_turns if prior_turns else None
        result = run(prompt, prior_context=prior_context)

        show_result(result)

        # Accumulate turns for context-aware follow-ups
        prior_turns.append({"author": "customer", "text": prompt, "inbound": True})
        prior_turns.append({"author": "XboxSupport", "text": result["reply"], "inbound": False})

        # Keep last 6 turns
        if len(prior_turns) > 6:
            prior_turns = prior_turns[-6:]


if __name__ == "__main__":
    main()
