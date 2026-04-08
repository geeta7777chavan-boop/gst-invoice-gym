from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckSpec:
    passed: bool
    message: str
    severity: str


@dataclass(frozen=True)
class InvoiceCase:
    invoice_id: str
    difficulty: str
    scenario: str
    visible_fields: dict[str, float]
    hidden_checks: dict[str, CheckSpec]
    compliance_issues: list[str]
    recommended_checks: list[str]
    correct_decision: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InvoiceCase":
        hidden_checks = {
            name: CheckSpec(**check_payload)
            for name, check_payload in payload["hidden_checks"].items()
        }
        return cls(
            invoice_id=payload["invoice_id"],
            difficulty=payload["difficulty"],
            scenario=payload["scenario"],
            visible_fields={
                key: float(value) for key, value in payload["visible_fields"].items()
            },
            hidden_checks=hidden_checks,
            compliance_issues=list(payload["compliance_issues"]),
            recommended_checks=list(payload["recommended_checks"]),
            correct_decision=payload["correct_decision"],
        )
