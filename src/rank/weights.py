from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Stage2Weights:
    tfidf: float
    amount: float
    keyword: float
    effort: float

    def __post_init__(self) -> None:
        for field_name in ("tfidf", "amount", "keyword", "effort"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"Stage2 weight '{field_name}' must be finite.")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Stage2 weight '{field_name}' must be between 0.0 and 1.0.")

    @classmethod
    def baseline(cls) -> Stage2Weights:
        return cls(tfidf=0.70, amount=0.20, keyword=0.10, effort=0.10)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> Stage2Weights:
        values = payload or {}
        baseline = cls.baseline()
        return cls(
            tfidf=float(values.get("tfidf", baseline.tfidf)),
            amount=float(values.get("amount", baseline.amount)),
            keyword=float(values.get("keyword", baseline.keyword)),
            effort=float(values.get("effort", baseline.effort)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "tfidf": self.tfidf,
            "amount": self.amount,
            "keyword": self.keyword,
            "effort": self.effort,
        }


@dataclass(frozen=True, slots=True)
class Stage3Weights:
    """`ev` weights `expected_value_norm` when the win model is enabled, else `ev_proxy_norm`."""

    stage2: float
    urgency: float
    ev: float

    def __post_init__(self) -> None:
        for field_name in ("stage2", "urgency", "ev"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"Stage3 weight '{field_name}' must be finite.")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Stage3 weight '{field_name}' must be between 0.0 and 1.0.")

        total = self.stage2 + self.urgency + self.ev
        if not math.isclose(total, 1.0, abs_tol=WEIGHT_TOLERANCE):
            raise ValueError(
                "Stage3 weights must sum to 1.0 "
                f"(received {total:.6f}, tolerance={WEIGHT_TOLERANCE})."
            )

    @classmethod
    def baseline(cls) -> Stage3Weights:
        return cls(stage2=0.80, urgency=0.15, ev=0.05)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> Stage3Weights:
        values = payload or {}
        baseline = cls.baseline()
        return cls(
            stage2=float(values.get("stage2", baseline.stage2)),
            urgency=float(values.get("urgency", baseline.urgency)),
            ev=float(values.get("ev", baseline.ev)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "stage2": self.stage2,
            "urgency": self.urgency,
            "ev": self.ev,
        }
