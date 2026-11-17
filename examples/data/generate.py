"""Deterministic synthetic datasets for examples and end-to-end tests (stdlib only).

    python3 examples/data/generate.py

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


def write(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    churn()
    houses()
