"""Deterministic synthetic datasets for examples and end-to-end tests (stdlib only).

    python3 examples/data/generate.py

segments.csv has no label (clustering); transactions.csv has a fraud label used only to evaluate the
anomaly detector, never to train it.

churn.csv contains a *planted leak* (`refund_amount` only exists after a customer churned)
so you can watch the audit catch it.
"""

import csv
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
        if fraud:
            amount = round(rng.uniform(800, 5000), 2)
            hour = rng.choice([1, 2, 3, 4])
            km = round(rng.uniform(300, 4000), 1)
            tries = rng.randint(2, 6)
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


def write(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    churn()
    houses()
    segments()
    transactions()
