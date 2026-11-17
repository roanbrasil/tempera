"""Algorithm registry. Adapters register themselves by name."""

from __future__ import annotations

from tempera_worker.algorithms.dummy import DummyAlgorithm
from tempera_worker.protocol import Algorithm

REGISTRY: dict[str, type[Algorithm]] = {
    DummyAlgorithm.name: DummyAlgorithm,
}


def get(name: str) -> type[Algorithm]:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise SystemExit(f"unknown algorithm `{name}` (known: {known})") from None
