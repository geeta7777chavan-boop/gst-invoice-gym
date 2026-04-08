from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState
    from ..task_definitions import (
        AVAILABLE_COMMANDS,
        CHECK_COMMANDS,
        DECISION_COMMANDS,
        TASKS,
        build_invoice_features,
        coverage_ratio,
        detection_ratio,
        empty_check_status,
        final_task_score,
        get_case,
        get_task,
        progress_score,
    )
except ImportError:
    from models import GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState
    from task_definitions import (
        AVAILABLE_COMMANDS,
        CHECK_COMMANDS,
        DECISION_COMMANDS,
        TASKS,
        build_invoice_features,
        coverage_ratio,
        detection_ratio,
        empty_check_status,
        final_task_score,
        get_case,
        get_task,
        progress_score,
    )


class GSTInvoiceGymEnvironment(
    Environment[GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState]
):
    """OpenEnv environment for GST invoice compliance training."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_id = "easy_invalid_supplier"
        self._current_task = TASKS[self._task_id]
        self._current_case = get_case(self._task_id)
        self._check_status = empty_check_status()
        self._completed_checks: set[str] = set()
        self._detected_issue_checks: set[str] = set()
        self._final_decision: str | None = None
        self._task_score = 0.0
        self._total_reward = 0.0
        self._last_feedback = "Reset the environment to start a task."
        self._state = GSTInvoiceState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_id,
            difficulty=self._current_task.difficulty,
            invoice_id=self._current_case.invoice_id,
            steps_remaining=self._current_task.max_steps,
            last_feedback=self._last_feedback,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "easy_invalid_supplier",
        case_id: str | None = None,
        **kwargs,
    ) -> GSTInvoiceObservation:
        del seed, kwargs
        self._task_id = task_id
        self._current_task = get_task(task_id)
        self._current_case = get_case(task_id, case_id)
        self._check_status = empty_check_status()
        self._completed_checks = set()
        self._detected_issue_checks = set()
        self._final_decision = None
        self._task_score = 0.0
        self._total_reward = 0.0
        self._last_feedback = self._current_task.objective
        self._state = GSTInvoiceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_id,
            difficulty=self._current_task.difficulty,
            invoice_id=self._current_case.invoice_id,
            completed_checks=[],
            detected_issue_checks=[],
            final_decision=None,
            task_score=0.0,
            total_reward=0.0,
            steps_remaining=self._current_task.max_steps,
            last_feedback=self._last_feedback,
        )
        return self._build_observation(reward=0.0, grader_score=0.0, done=False)

    def step(
        self,
        action: GSTInvoiceAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> GSTInvoiceObservation:
        del timeout_s, kwargs
        if self._state.episode_id is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        grader_score = 0.0
        reward = 0.0
        done = False
        metadata: dict[str, object] = {
            "task_id": self._task_id,
            "invoice_id": self._current_case.invoice_id,
            "command": action.command,
        }

        if action.command in CHECK_COMMANDS:
            check_name = CHECK_COMMANDS[action.command]
            check_spec = self._current_case.hidden_checks[check_name]

            if check_name in self._completed_checks:
                self._last_feedback = f"{check_name} was already checked."
                metadata["grader"] = {
                    "type": "check",
                    "check_name": check_name,
                    "status": "duplicate",
                }
            else:
                self._completed_checks.add(check_name)
                self._check_status[check_name] = "pass" if check_spec.passed else "fail"
                if not check_spec.passed:
                    self._detected_issue_checks.add(check_name)

                recommended = check_name in self._current_case.recommended_checks
                if not check_spec.passed and recommended:
                    reward = 0.60
                    grader_score = 1.00
                elif not check_spec.passed:
                    reward = 0.50
                    grader_score = 0.85
                elif recommended:
                    reward = 0.25
                    grader_score = 0.55
                else:
                    reward = 0.10
                    grader_score = 0.25

                self._last_feedback = check_spec.message
                metadata["grader"] = {
                    "type": "check",
                    "check_name": check_name,
                    "status": "pass" if check_spec.passed else "fail",
                    "severity": check_spec.severity,
                    "recommended": recommended,
                }

            self._task_score = progress_score(
                self._current_case,
                self._completed_checks,
                self._detected_issue_checks,
            )
        else:
            decision = DECISION_COMMANDS[action.command]
            self._final_decision = decision
            grader_score = 1.0 if decision == self._current_case.correct_decision else 0.0
            self._task_score = final_task_score(
                self._current_case,
                self._completed_checks,
                self._detected_issue_checks,
                decision,
            )
            reward = self._task_score
            done = True
            coverage = coverage_ratio(self._current_case, self._completed_checks)
            detection = detection_ratio(
                self._current_case,
                self._detected_issue_checks,
            )
            self._last_feedback = (
                f"Decision '{decision}' evaluated against expected outcome "
                f"'{self._current_case.correct_decision}'."
            )
            metadata["grader"] = {
                "type": "decision",
                "decision": decision,
                "expected_decision": self._current_case.correct_decision,
                "decision_correct": grader_score == 1.0,
                "coverage": coverage,
                "issue_detection": detection,
                "task_score": self._task_score,
            }

        if not done and self._state.step_count >= self._current_task.max_steps:
            done = True
            grader_score = 0.0
            reward = round(min(0.20, self._task_score * 0.20), 2)
            self._task_score = reward
            self._last_feedback = (
                "Episode terminated because the agent did not make a final "
                "approve, reject, or flag decision in time."
            )
            metadata["grader"] = {
                "type": "timeout",
                "coverage": coverage_ratio(self._current_case, self._completed_checks),
                "issue_detection": detection_ratio(
                    self._current_case,
                    self._detected_issue_checks,
                ),
            }

        self._total_reward = round(self._total_reward + reward, 2)
        self._state.completed_checks = sorted(self._completed_checks)
        self._state.detected_issue_checks = sorted(self._detected_issue_checks)
        self._state.final_decision = self._final_decision
        self._state.task_score = self._task_score
        self._state.total_reward = self._total_reward
        self._state.steps_remaining = max(
            0, self._current_task.max_steps - self._state.step_count
        )
        self._state.last_feedback = self._last_feedback

        return self._build_observation(
            reward=reward,
            grader_score=grader_score,
            done=done,
            metadata=metadata,
        )

    @property
    def state(self) -> GSTInvoiceState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="gst_invoice_gym",
            description=(
                "GST invoice processing environment with three graded compliance "
                "tasks for approve, reject, and manual review decisions."
            ),
            version="0.1.0",
            author="OpenEnv GST Invoice Gym",
        )

    def _build_observation(
        self,
        *,
        reward: float,
        grader_score: float,
        done: bool,
        metadata: dict[str, object] | None = None,
    ) -> GSTInvoiceObservation:
        return GSTInvoiceObservation(
            task_id=self._task_id,
            difficulty=self._current_task.difficulty,
            invoice_id=self._current_case.invoice_id,
            scenario=self._current_case.scenario,
            objective=self._current_task.objective,
            invoice_features=build_invoice_features(self._current_case),
            check_status=dict(self._check_status),
            recommended_checks=list(self._current_case.recommended_checks),
            available_actions=list(AVAILABLE_COMMANDS),
            compliance_issues_found=sorted(self._detected_issue_checks),
            last_feedback=self._last_feedback,
            grader_score=grader_score,
            task_score=self._task_score,
            final_decision=self._final_decision,
            steps_remaining=max(0, self._current_task.max_steps - self._state.step_count),
            done=done,
            reward=reward,
            metadata=metadata or {},
        )
