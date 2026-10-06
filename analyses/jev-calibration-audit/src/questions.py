"""The pinned question templates. One place, so every experiment asks the same way."""
from __future__ import annotations

STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def choice_q(instructions: str, criteria: dict[str, str]) -> dict:
    return {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}


def intent_q(criteria: dict[str, str]) -> dict:
    return choice_q("Which intent does the customer message express?", criteria)


def stars_q() -> dict:
    return {"q": {"type": "score",
                  "instructions": "How many stars did the reviewer give this product?",
                  "criteria": STARS}}


def noul_q(question: str) -> dict:
    return {"q": {"type": "noul", "instructions": question}}
