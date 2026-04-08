from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ALL_CHECKS = (
    "supplier_identity",
    "buyer_identity",
    "tax_regime",
    "tax_math",
    "mandatory_fields",
)

CHECK_COMMANDS = {
    "check_supplier_identity": "supplier_identity",
    "check_buyer_identity": "buyer_identity",
    "check_tax_regime": "tax_regime",
    "check_tax_math": "tax_math",
    "check_mandatory_fields": "mandatory_fields",
}

DECISION_COMMANDS = {
    "approve": "approve",
    "reject": "reject",
    "flag_for_review": "flag",
}

AVAILABLE_COMMANDS = list(CHECK_COMMANDS) + list(DECISION_COMMANDS)


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


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    title: str
    objective: str
    grader_description: str
    default_case_id: str
    max_steps: int


TASKS = {
    "easy_invalid_supplier": TaskSpec(
        task_id="easy_invalid_supplier",
        difficulty="easy",
        title="Easy Supplier Identity Rejection",
        objective=(
            "Inspect the invoice for a single obvious compliance issue and choose "
            "approve or reject with minimal wasted steps."
        ),
        grader_description=(
            "Scores the agent on detecting the invalid supplier identity, covering "
            "recommended checks, and selecting reject."
        ),
        default_case_id="GST-002",
        max_steps=4,
    ),
    "medium_tax_regime_mismatch": TaskSpec(
        task_id="medium_tax_regime_mismatch",
        difficulty="medium",
        title="Medium Tax Regime Mismatch",
        objective=(
            "Detect an interstate GST tax regime mismatch, verify tax math, and "
            "make the correct rejection decision."
        ),
        grader_description=(
            "Scores the agent on surfacing the wrong tax regime, validating tax "
            "math, and choosing reject."
        ),
        default_case_id="GST-003",
        max_steps=5,
    ),
    "hard_manual_review_needed": TaskSpec(
        task_id="hard_manual_review_needed",
        difficulty="hard",
        title="Hard Manual Review Escalation",
        objective=(
            "Inspect a borderline invoice, identify why manual review is needed, "
            "and flag it instead of over-approving or auto-rejecting."
        ),
        grader_description=(
            "Scores the agent on finding the buyer identity discrepancy, covering "
            "recommended checks, and selecting flag for review."
        ),
        default_case_id="GST-005",
        max_steps=6,
    ),
}


def _data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "invoices.json"


def load_cases() -> dict[str, InvoiceCase]:
    payload = json.loads(_data_path().read_text(encoding="utf-8"))
    cases: dict[str, InvoiceCase] = {}
    for raw_case in payload:
        cases[raw_case["invoice_id"]] = InvoiceCase(
            invoice_id=raw_case["invoice_id"],
            difficulty=raw_case["difficulty"],
            scenario=raw_case["scenario"],
            visible_fields={
                key: float(value) for key, value in raw_case["visible_fields"].items()
            },
            hidden_checks={
                key: CheckSpec(**value)
                for key, value in raw_case["hidden_checks"].items()
            },
            compliance_issues=list(raw_case["compliance_issues"]),
            recommended_checks=list(raw_case["recommended_checks"]),
            correct_decision=raw_case["correct_decision"],
        )
    return cases


CASES = load_cases()


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid task ids: {sorted(TASKS)}"
        )
    return TASKS[task_id]


def get_case(task_id: str, case_id: str | None = None) -> InvoiceCase:
    task = get_task(task_id)
    resolved_case_id = case_id or task.default_case_id
    if resolved_case_id not in CASES:
        raise ValueError(f"Unknown case_id '{resolved_case_id}'.")
    return CASES[resolved_case_id]


def empty_check_status() -> dict[str, str]:
    return {check_name: "unknown" for check_name in ALL_CHECKS}


def build_invoice_features(case: InvoiceCase) -> dict[str, float]:
    return {
        **case.visible_fields,
        "difficulty_code": {"easy": 0.0, "medium": 1.0, "hard": 2.0}[case.difficulty],
    }


def failing_checks(case: InvoiceCase) -> list[str]:
    return [
        check_name
        for check_name, check_spec in case.hidden_checks.items()
        if not check_spec.passed
    ]


def detection_ratio(case: InvoiceCase, detected_checks: set[str]) -> float:
    failing = failing_checks(case)
    if not failing:
        return 1.0
    detected = len(set(failing) & detected_checks)
    return round(detected / len(failing), 2)


def coverage_ratio(case: InvoiceCase, completed_checks: set[str]) -> float:
    required = set(case.recommended_checks)
    if not required:
        return 1.0
    covered = len(required & completed_checks)
    return round(covered / len(required), 2)


def progress_score(
    case: InvoiceCase,
    completed_checks: set[str],
    detected_checks: set[str],
) -> float:
    coverage = coverage_ratio(case, completed_checks)
    detection = detection_ratio(case, detected_checks)
    return round(min(1.0, 0.55 * coverage + 0.45 * detection), 2)


def final_task_score(
    case: InvoiceCase,
    completed_checks: set[str],
    detected_checks: set[str],
    decision: str | None,
) -> float:
    coverage = coverage_ratio(case, completed_checks)
    detection = detection_ratio(case, detected_checks)
    decision_score = 1.0 if decision == case.correct_decision else 0.0
    score = 0.60 * decision_score + 0.25 * coverage + 0.15 * detection
    if decision == "approve" and case.correct_decision != "approve":
        score -= 0.10
    return round(max(0.0, min(1.0, score)), 2)


def task_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for task in TASKS.values():
        catalog.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "title": task.title,
                "objective": task.objective,
                "grader_description": task.grader_description,
                "default_case_id": task.default_case_id,
                "max_steps": task.max_steps,
                "available_actions": AVAILABLE_COMMANDS,
            }
        )
    return catalog
