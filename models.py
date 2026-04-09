from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


SupportedCommand = Literal[
    "check_supplier_identity",
    "check_buyer_identity",
    "check_tax_regime",
    "check_tax_math",
    "check_mandatory_fields",
    "approve",
    "reject",
    "flag_for_review",
]


class GSTInvoiceAction(Action):
    """Typed action model for the GST invoice environment."""

    command: SupportedCommand = Field(
        ...,
        description="One of the supported inspection or decision commands.",
    )
    notes: str = Field(
        default="",
        description="Optional agent rationale for auditability.",
    )


class GSTInvoiceObservation(Observation):
    """Typed observation model returned after reset() and step()."""

    task_id: str = Field(..., description="Current task identifier.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Difficulty tier for the current task.",
    )
    invoice_id: str = Field(..., description="Synthetic invoice identifier.")
    scenario: str = Field(..., description="Short description of the invoice scenario.")
    objective: str = Field(..., description="Task objective for the current episode.")
    invoice_features: dict[str, float] = Field(
        default_factory=dict,
        description="Structured invoice attributes visible to the agent.",
    )
    check_status: dict[str, str] = Field(
        default_factory=dict,
        description="Status per compliance check: unknown, pass, or fail.",
    )
    recommended_checks: list[str] = Field(
        default_factory=list,
        description="Checks the grader expects the agent to prioritize.",
    )
    available_actions: list[str] = Field(
        default_factory=list,
        description="Allowed action names for the current episode.",
    )
    compliance_issues_found: list[str] = Field(
        default_factory=list,
        description="Failing checks surfaced so far by the agent.",
    )
    last_feedback: str = Field(
        default="",
        description="Most recent grader or environment feedback.",
    )
    grader_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic score assigned to the latest action.",
    )
    task_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Current task completion score in the open interval (0.0, 1.0). "
            "The environment uses 0.01 as the minimum active score and 0.99 "
            "as the maximum perfect score."
        ),
    )
    final_decision: str | None = Field(
        default=None,
        description="Final approve, reject, or flag decision once chosen.",
    )
    steps_remaining: int = Field(
        default=0,
        ge=0,
        description="Number of remaining steps before the episode terminates.",
    )


class GSTInvoiceState(State):
    """Typed environment state exposed through the OpenEnv state() endpoint."""

    task_id: str | None = Field(default=None, description="Current task identifier.")
    difficulty: str | None = Field(default=None, description="Current task difficulty.")
    invoice_id: str | None = Field(default=None, description="Current invoice identifier.")
    completed_checks: list[str] = Field(
        default_factory=list,
        description="Checks already executed in the current episode.",
    )
    detected_issue_checks: list[str] = Field(
        default_factory=list,
        description="Failing checks discovered so far.",
    )
    final_decision: str | None = Field(
        default=None,
        description="Final decision if one has been made.",
    )
    task_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Current normalized task score in the open interval (0.0, 1.0)."
        ),
    )
    total_reward: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative reward accrued in the episode.",
    )
    steps_remaining: int = Field(
        default=0,
        ge=0,
        description="How many actions remain before the episode ends.",
    )
    last_feedback: str = Field(
        default="",
        description="Most recent grader feedback stored in state.",
    )
