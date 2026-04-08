from __future__ import annotations

import json
from pathlib import Path

from env.models import InvoiceCase

CHECK_NAMES = (
    "supplier_identity",
    "buyer_identity",
    "tax_regime",
    "tax_math",
    "mandatory_fields",
)

CHECK_ACTIONS = {
    0: "supplier_identity",
    1: "buyer_identity",
    2: "tax_regime",
    3: "tax_math",
    4: "mandatory_fields",
}

DECISION_ACTIONS = {
    5: "approve",
    6: "reject",
    7: "flag",
}

ACTION_LABELS = {
    0: "check_supplier_identity",
    1: "check_buyer_identity",
    2: "check_tax_regime",
    3: "check_tax_math",
    4: "check_mandatory_fields",
    5: "approve",
    6: "reject",
    7: "flag_for_review",
}

DIFFICULTY_TO_CODE = {
    "easy": 0.0,
    "medium": 1.0,
    "hard": 2.0,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return project_root() / "data" / "invoices.json"


def load_invoice_cases(data_path: str | Path | None = None) -> list[InvoiceCase]:
    resolved_path = Path(data_path) if data_path is not None else default_data_path()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return [InvoiceCase.from_dict(case) for case in payload]


def build_feature_vector(case: InvoiceCase) -> list[float]:
    fields = case.visible_fields
    return [
        fields["taxable_amount"],
        fields["declared_cgst"],
        fields["declared_sgst"],
        fields["declared_igst"],
        fields["line_item_count"],
        fields["invoice_age_days"],
        fields["supplier_state_code"],
        fields["buyer_state_code"],
        fields["total_amount"],
        DIFFICULTY_TO_CODE[case.difficulty],
    ]


def failing_checks(case: InvoiceCase) -> list[str]:
    return [
        check_name
        for check_name, check_spec in case.hidden_checks.items()
        if not check_spec.passed
    ]


def detected_issues(case: InvoiceCase, completed_checks: set[str]) -> list[str]:
    found: list[str] = []
    for check_name in completed_checks:
        check_spec = case.hidden_checks[check_name]
        if not check_spec.passed:
            found.append(check_name)
    return sorted(found)


def action_name(action: int) -> str:
    if action in ACTION_LABELS:
        return ACTION_LABELS[action]
    raise ValueError(f"Unsupported action: {action}")
