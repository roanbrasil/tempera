"""Deterministic synthetic datasets for examples and end-to-end tests (stdlib only).

    python3 examples/data/generate.py

segments.csv has no label (clustering); transactions.csv has a fraud label used only to evaluate the
anomaly detector, never to train it.

agent_traces.jsonl holds support-agent conversations with tool calls (OpenAI and Hermes formats)
and *planted problems* for the trace audit: personal data, malformed tool calls, prompts copied
across splits and overlap with agent_benchmark.jsonl. agent_traces_clean.jsonl is the same
data without them.

churn.csv contains a *planted leak* (`refund_amount` only exists after a customer churned)
so you can watch the audit catch it.
"""

import csv
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent


def churn(n: int = 3000) -> None:
    rng = random.Random(7)
    rows = []
    for i in range(n):
        plan = rng.choice(["basic", "basic", "pro", "team"])
        tenure = rng.randint(1, 72)
        fee = {"basic": 19, "pro": 49, "team": 99}[plan] + rng.choice([0, 0, 5, 10])
        tickets = min(int(rng.expovariate(0.6)), 12)
        discount = rng.random() < 0.3
        logit = -1.2 - 0.045 * tenure + 0.35 * tickets + 0.015 * fee - 0.8 * discount + rng.gauss(0, 0.6)
        churned = rng.random() < 1 / (1 + math.exp(-logit))
        refund = round(rng.uniform(5, fee), 2) if churned and rng.random() < 0.97 else 0.0
        month = 1 + i * 24 // n
        rows.append({
            "customer_id": f"C{100000 + i}",
            "signup_date": f"{2023 + (month - 1) // 12}-{(month - 1) % 12 + 1:02d}-{rng.randint(1, 28):02d}",
            "tenure_months": tenure,
            "plan": plan,
            "monthly_fee": fee,
            "support_tickets": tickets,
            "has_discount": str(discount).lower(),
            "country": "BR",
            "refund_amount": refund,
            "churned": "yes" if churned else "no",
        })
    write("churn.csv", rows)


def houses(n: int = 2000) -> None:
    rng = random.Random(11)
    hoods = {"centro": 1.35, "jardins": 1.6, "vila_nova": 1.0, "industrial": 0.75}
    rows = []
    for _ in range(n):
        hood = rng.choice(list(hoods))
        area = round(rng.uniform(35, 260), 1)
        bedrooms = max(1, min(6, int(area // 45) + rng.choice([-1, 0, 0, 1])))
        age = rng.randint(0, 60)
        garage = rng.random() < 0.6
        price = (2400 * area + 18000 * bedrooms + 25000 * garage - 900 * age) * hoods[hood]
        price *= math.exp(rng.gauss(0, 0.08))
        rows.append({
            "area_m2": area,
            "bedrooms": bedrooms,
            "neighborhood": hood,
            "age_years": age,
            "has_garage": str(garage).lower(),
            "price": round(price, -2),
        })
    write("houses.csv", rows)


def segments(n: int = 1500) -> None:
    """Customers from 3 behavioral segments (no label): for clustering."""
    rng = random.Random(23)
    centers = {
        "bargain": (25, 2, 0.8),  # avg ticket, visits/month, discount usage
        "loyal": (60, 8, 0.2),
        "premium": (220, 3, 0.05),
    }
    rows = []
    for i in range(n):
        seg = list(centers)[i % 3]
        ticket, visits, disc = centers[seg]
        rows.append({
            "avg_ticket": round(max(5, rng.gauss(ticket, ticket * 0.15)), 2),
            "visits_per_month": max(0, round(rng.gauss(visits, 1.2), 1)),
            "discount_share": round(min(1, max(0, rng.gauss(disc, 0.07))), 3),
            "channel": rng.choice(["app", "web"] if seg != "bargain" else ["app", "web", "store"]),
        })
    write("segments.csv", rows)


def transactions(n: int = 4000) -> None:
    """Card transactions with ~1.5% planted fraud; the label is for evaluation only."""
    rng = random.Random(31)
    rows = []
    for _ in range(n):
        fraud = rng.random() < 0.015
        if fraud and rng.random() < 0.4:
            # Camouflaged fraud: normal amount and hour, only one odd signal.
            amount = round(math.exp(rng.gauss(3.8, 0.8)), 2)
            hour = int(min(23, max(0, rng.gauss(14, 4))))
            km = round(rng.uniform(40, 600), 1)
            tries = rng.choice([1, 2])
        elif fraud:
            amount = round(rng.uniform(300, 5000), 2)
            hour = rng.choice([0, 1, 2, 3, 4, 22, 23])
            km = round(rng.uniform(100, 4000), 1)
            tries = rng.randint(1, 6)
        else:
            amount = round(math.exp(rng.gauss(3.5, 0.8)), 2)
            hour = int(min(23, max(0, rng.gauss(14, 4))))
            km = round(abs(rng.gauss(5, 8)), 1)
            tries = 1 if rng.random() < 0.97 else 2
        rows.append({
            "amount": amount,
            "hour": hour,
            "km_from_home": km,
            "attempts": tries,
            "merchant": rng.choice(["grocery", "fuel", "online", "travel", "restaurant"]),
            "is_fraud": "fraud" if fraud else "normal",
        })
    write("transactions.csv", rows)


AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Status and ETA of an order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "refund_order", "description": "Refund part or all of an order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "amount": {"type": "number"},
            "reason": {"type": "string", "enum": ["damaged", "late", "wrong_item", "other"]}},
            "required": ["order_id", "amount", "reason"]}}},
    {"type": "function", "function": {
        "name": "search_faq", "description": "Search the help center.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "handoff_to_human", "description": "Escalate to a human agent.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
]

_OPEN = ["hi", "hello", "good morning", "hey there", "excuse me", "quick question", "hi team", "ola"]
_ITEMS = ["blender", "headphones", "running shoes", "phone case", "desk lamp", "backpack", "coffee maker",
          "keyboard", "yoga mat", "water bottle", "monitor stand", "air fryer", "jacket", "router"]
_FAQ = ["how do I change my delivery address", "what is the return window for electronics",
        "can I pay with pix", "do you ship to portugal", "how do I reset my password",
        "is there a warranty on refurbished items", "how do I cancel my subscription",
        "can I split a payment between two cards", "where do I find my invoice",
        "how long does a refund take to show up"]
_CHAT = ["thanks, that is all", "what are your opening hours", "you are great, bye",
         "can you tell me a joke", "who are you", "never mind, I figured it out"]


def _conv(rng: random.Random, i: int) -> tuple[str, list | None, str]:
    """(user prompt, tool call or None, final answer)."""
    order = f"{rng.randint(10, 99)}-{rng.randint(100000, 999999)}"
    item = rng.choice(_ITEMS)
    hi = rng.choice(_OPEN)
    kind = rng.random()
    if kind < 0.30:
        tail = rng.choice(["it has been a while", "the tracking page is blank", "I need it by friday",
                           "the app shows nothing", "it said two days", f"it is a gift for my {rng.choice(['mom', 'son', 'boss', 'friend'])}"])
        return (f"{hi}, where is my {item}? order {order}, {tail}",
                ["get_order_status", {"order_id": order}], f"Order {order} is in transit.")
    if kind < 0.50:
        reason = rng.choice(["damaged", "late", "wrong_item"])
        amount = round(rng.uniform(15, 400), 2)
        why = {"damaged": f"the {item} arrived broken", "late": f"the {item} came {rng.randint(3, 20)} days late",
               "wrong_item": f"I got a {rng.choice(_ITEMS)} instead of a {item}"}[reason]
        return (f"{hi}, {why}. order {order}, I paid {amount}, I want my money back",
                ["refund_order", {"order_id": order, "amount": amount, "reason": reason}],
                f"Refund of {amount} issued for order {order}.")
    if kind < 0.68:
        q = rng.choice(_FAQ)
        return (f"{hi}, {q}? asking about my {item} (ref {rng.randint(1000, 9999)})",
                ["search_faq", {"query": q}], "Here is what the help center says.")
    if kind < 0.78:
        return (f"{hi}, I have asked {rng.randint(2, 6)} times about order {order} and nobody helps, "
                f"I want to talk to a person about the {item}",
                ["handoff_to_human", {"reason": "repeated unresolved contact"}], "Connecting you to a human.")
    # No tool: the right answer is to just reply.
    return (f"{rng.choice(_CHAT)} ({hi}, ticket {rng.randint(10000, 99999)})", None, "Happy to help!")


def _record(rng: random.Random, i: int, prompt: str, call: list | None, answer: str,
            raw_args: str | None = None) -> dict:
    hermes = i % 10 >= 7
    args = raw_args if raw_args is not None else (json.dumps(call[1]) if call else None)
    if hermes:
        tools = json.dumps(AGENT_TOOLS)
        msgs = [{"from": "system", "value": f"You are a support agent. <tools> {tools} </tools>"},
                {"from": "human", "value": prompt}]
        if call:
            msgs += [{"from": "gpt", "value": f'<tool_call>\n{{"name": "{call[0]}", "arguments": {args}}}\n</tool_call>'},
                     {"from": "tool", "value": "<tool_response>{\"ok\": true}</tool_response>"}]
        msgs.append({"from": "gpt", "value": answer})
        return {"id": f"conv-{i:05d}", "conversations": msgs}
    msgs = [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": prompt}]
    if call:
        msgs += [{"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{i}", "type": "function", "function": {"name": call[0], "arguments": args}}]},
                 {"role": "tool", "tool_call_id": f"call_{i}", "content": "{\"ok\": true}"}]
    msgs.append({"role": "assistant", "content": answer})
    return {"id": f"conv-{i:05d}", "messages": msgs, "tools": AGENT_TOOLS}


def agent_traces(n: int = 800) -> None:
    rng = random.Random(41)
    # A clean dataset has no two prompts that differ only in their numbers: those are
    # near duplicates and the audit (rightly) flags them across splits.
    base, seen = [], set()
    while len(base) < n:
        c = _conv(rng, len(base))
        key = "".join(ch for ch in c[0] if not ch.isdigit())
        if key not in seen:
            seen.add(key)
            base.append(c)
    clean = [_record(rng, i, *c) for i, c in enumerate(base)]
    bench = [f"{rng.choice(_OPEN)}, {rng.choice(_FAQ)} for an order placed abroad with store credit {k}"
             for k in range(40)]

    planted = []
    for i, (prompt, call, answer) in enumerate(base):
        raw = None
        if i % 33 == 5:  # personal data in ~3% of conversations
            prompt += f" my email is {rng.choice(['ana', 'joao', 'li', 'sam'])}.{i}@example.com"
        if i % 40 == 9 and call:  # malformed calls: broken JSON or wrong types
            raw = '{"order_id": ' if call[0] != "refund_order" else json.dumps({**call[1], "amount": str(call[1]["amount"])})
        planted.append(_record(rng, i, prompt, call, answer, raw))
    for k in range(40):  # same prompt re-logged under a new id (lands in other splits)
        prompt, call, answer = base[k * 7]
        planted.append(_record(rng, n + k, prompt, call, answer))
    for k in range(8):  # benchmark items that leaked into the training data
        planted.append(_record(rng, n + 100 + k, bench[k], ["search_faq", {"query": bench[k]}], "See the FAQ."))

    for name, rows in [("agent_traces.jsonl", planted), ("agent_traces_clean.jsonl", clean),
                       ("agent_benchmark.jsonl", [{"prompt": b} for b in bench])]:
        with (HERE / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    agent_traces()
    churn()
    houses()
    segments()
    transactions()
