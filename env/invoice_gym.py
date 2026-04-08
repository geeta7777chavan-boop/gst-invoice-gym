from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.models import InvoiceCase
from env.utils import (
    ACTION_LABELS,
    CHECK_ACTIONS,
    CHECK_NAMES,
    DECISION_ACTIONS,
    action_name,
    build_feature_vector,
    detected_issues,
    failing_checks,
    load_invoice_cases,
)


class GSTInvoiceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path: str | Path | None = None, max_steps: int = 6) -> None:
        super().__init__()
        self.data_path = Path(data_path) if data_path is not None else None
        self.max_steps = max_steps
        self.cases = load_invoice_cases(self.data_path)
        self.case_lookup = {case.invoice_id: case for case in self.cases}
        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Dict(
            {
                "invoice_features": spaces.Box(
                    low=0.0,
                    high=1_000_000.0,
                    shape=(10,),
                    dtype=np.float32,
                ),
                "check_status": spaces.MultiDiscrete([3] * len(CHECK_NAMES)),
                "remaining_steps": spaces.Discrete(self.max_steps + 1),
            }
        )
        self._rng = np.random.default_rng()
        self.current_case: InvoiceCase | None = None
        self.completed_checks: set[str] = set()
        self.check_status = np.zeros(len(CHECK_NAMES), dtype=np.int64)
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        case_id = (options or {}).get("case_id")
        if case_id is not None:
            if case_id not in self.case_lookup:
                raise ValueError(f"Unknown case_id '{case_id}'.")
            self.current_case = self.case_lookup[case_id]
        else:
            case_index = int(self._rng.integers(0, len(self.cases)))
            self.current_case = self.cases[case_index]

        self.completed_checks = set()
        self.check_status = np.zeros(len(CHECK_NAMES), dtype=np.int64)
        self.step_count = 0

        return self._observation(), self._reset_info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.current_case is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space.")

        self.step_count += 1
        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "invoice_id": self.current_case.invoice_id,
            "action_name": action_name(action),
        }

        if action in CHECK_ACTIONS:
            reward = self._handle_check_action(CHECK_ACTIONS[action], info)
        else:
            reward = self._handle_decision_action(DECISION_ACTIONS[action], info)
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            reward -= 1.5
            info["truncation_reason"] = "max_steps_reached_without_final_decision"

        return self._observation(), float(reward), terminated, truncated, info

    def state(self) -> dict[str, Any]:
        if self.current_case is None:
            return {
                "episode_started": False,
                "step_count": 0,
                "completed_checks": [],
            }
        return {
            "episode_started": True,
            "invoice_id": self.current_case.invoice_id,
            "step_count": self.step_count,
            "completed_checks": sorted(self.completed_checks),
            "remaining_steps": self.max_steps - self.step_count,
        }

    def render(self) -> str:
        if self.current_case is None:
            return "Environment not initialized."
        return (
            f"Invoice {self.current_case.invoice_id} | "
            f"difficulty={self.current_case.difficulty} | "
            f"step={self.step_count}/{self.max_steps}"
        )

    def close(self) -> None:
        self.current_case = None
        self.completed_checks = set()
        self.check_status = np.zeros(len(CHECK_NAMES), dtype=np.int64)
        self.step_count = 0

    def _handle_check_action(self, check_name: str, info: dict[str, Any]) -> float:
        assert self.current_case is not None
        base_step_penalty = -0.05
        check_index = CHECK_NAMES.index(check_name)

        if check_name in self.completed_checks:
            info["grader"] = {
                "check_name": check_name,
                "status": "duplicate",
                "message": "This compliance check was already run.",
            }
            return base_step_penalty - 0.15

        self.completed_checks.add(check_name)
        check_spec = self.current_case.hidden_checks[check_name]
        self.check_status[check_index] = 1 if check_spec.passed else 2

        info["grader"] = {
            "check_name": check_name,
            "status": "pass" if check_spec.passed else "fail",
            "severity": check_spec.severity,
            "message": check_spec.message,
        }

        if not check_spec.passed:
            return base_step_penalty + 0.45
        if check_name in self.current_case.recommended_checks:
            return base_step_penalty + 0.15
        return base_step_penalty + 0.05

    def _handle_decision_action(self, decision: str, info: dict[str, Any]) -> float:
        assert self.current_case is not None
        expected_decision = self.current_case.correct_decision
        found_checks = detected_issues(self.current_case, self.completed_checks)
        required_checks = set(self.current_case.recommended_checks)
        coverage = 1.0 if not required_checks else len(required_checks & self.completed_checks) / len(required_checks)

        decision_correct = decision == expected_decision
        info["grader"] = {
            "decision": decision,
            "expected_decision": expected_decision,
            "decision_correct": decision_correct,
            "inspection_coverage": round(coverage, 2),
            "detected_issue_checks": found_checks,
            "undetected_issue_checks": sorted(set(failing_checks(self.current_case)) - set(found_checks)),
        }

        if decision_correct:
            efficiency_bonus = max(0.0, 0.6 - 0.1 * (self.step_count - 1))
            coverage_bonus = 0.4 * coverage
            return 2.8 + efficiency_bonus + coverage_bonus

        penalty = -2.8
        if decision == "approve" and expected_decision != "approve":
            penalty -= 0.5
        if decision == "reject" and expected_decision == "flag":
            penalty -= 0.25
        return penalty

    def _observation(self) -> dict[str, Any]:
        assert self.current_case is not None
        return {
            "invoice_features": np.array(
                build_feature_vector(self.current_case),
                dtype=np.float32,
            ),
            "check_status": self.check_status.copy(),
            "remaining_steps": self.max_steps - self.step_count,
        }

    def _reset_info(self) -> dict[str, Any]:
        assert self.current_case is not None
        return {
            "invoice_id": self.current_case.invoice_id,
            "difficulty": self.current_case.difficulty,
            "scenario": self.current_case.scenario,
            "task": (
                "Run targeted GST compliance checks, then choose approve, reject, "
                "or flag for review."
            ),
            "action_meanings": ACTION_LABELS,
        }
