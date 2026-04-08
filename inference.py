from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

from client import GSTInvoiceGymEnv
from models import GSTInvoiceAction, GSTInvoiceObservation
from task_definitions import AVAILABLE_COMMANDS, CHECK_COMMANDS, TASKS

DEFAULT_ENV_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LEGACY_HF_TOKEN = os.environ.get("HF_TOKEN")
API_KEY = OPENAI_API_KEY or LEGACY_HF_TOKEN
MAX_STEPS = 6
TEMPERATURE = 0.0


@dataclass
class TaskRun:
    task_id: str
    score: float
    steps: int
    final_decision: str | None


def print_task_header(task_id: str, invoice_id: str, difficulty: str) -> None:
    print()
    print("=" * 72)
    print(f"Task: {task_id}")
    print(f"Invoice: {invoice_id} | Difficulty: {difficulty}")
    print("-" * 72)


def print_step_line(
    *,
    step_index: int,
    command: str,
    reward: float,
    grader_score: float,
    task_score: float,
    done: bool,
    policy: str,
) -> None:
    print(
        f"Step {step_index:>2} | "
        f"action={command:<23} "
        f"reward={reward:>4.2f} "
        f"grader={grader_score:>4.2f} "
        f"task={task_score:>4.2f} "
        f"done={'yes' if done else 'no ':<3} "
        f"policy={policy}"
    )


def print_task_footer(score: float, steps: int, final_decision: str | None) -> None:
    print("-" * 72)
    print(
        f"Result: score={score:.2f} | steps={steps} | "
        f"final_decision={final_decision or 'none'}"
    )


def print_summary(runs: list[TaskRun], average_score: float) -> None:
    print()
    print("=" * 72)
    print("Summary")
    print("-" * 72)
    print(f"{'Task':<32} {'Score':>7} {'Steps':>7} {'Decision':>16}")
    print("-" * 72)
    for run in runs:
        print(
            f"{run.task_id:<32} "
            f"{run.score:>7.2f} "
            f"{run.steps:>7} "
            f"{(run.final_decision or 'none'):>16}"
        )
    print("-" * 72)
    print(f"{'Average score':<32} {average_score:>7.2f}")
    print("=" * 72)


def build_client() -> OpenAI | None:
    if not API_BASE_URL or not MODEL_NAME or not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def parse_command(text: str) -> str | None:
    normalized = text.strip()
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            command = payload.get("command")
            if command in AVAILABLE_COMMANDS:
                return command
    except json.JSONDecodeError:
        pass

    lowered = normalized.lower()
    for command in AVAILABLE_COMMANDS:
        if command in lowered:
            return command
    return None


def fallback_command(observation: GSTInvoiceObservation) -> str:
    failing_checks = [
        check_name
        for check_name, status in observation.check_status.items()
        if status == "fail"
    ]
    remaining_recommended = [
        check_name
        for check_name in observation.recommended_checks
        if observation.check_status.get(check_name) == "unknown"
    ]

    if remaining_recommended:
        for command, check_name in CHECK_COMMANDS.items():
            if check_name == remaining_recommended[0]:
                return command

    if failing_checks:
        if observation.task_id == "hard_manual_review_needed":
            return "flag_for_review"
        return "reject"

    if observation.task_id == "hard_manual_review_needed":
        return "flag_for_review"
    return "approve"


def model_command(
    client: OpenAI | None,
    observation: GSTInvoiceObservation,
) -> tuple[str, str]:
    if client is None:
        return fallback_command(observation), "deterministic_baseline"

    prompt = {
        "task_id": observation.task_id,
        "difficulty": observation.difficulty,
        "objective": observation.objective,
        "invoice_id": observation.invoice_id,
        "scenario": observation.scenario,
        "invoice_features": observation.invoice_features,
        "check_status": observation.check_status,
        "recommended_checks": observation.recommended_checks,
        "compliance_issues_found": observation.compliance_issues_found,
        "steps_remaining": observation.steps_remaining,
        "available_actions": observation.available_actions,
        "last_feedback": observation.last_feedback,
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are controlling a GST invoice compliance environment. "
                        "Return only JSON in the form "
                        '{"command":"<one-supported-command>","reason":"short reason"}'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                },
            ],
        )
        response_text = completion.choices[0].message.content or ""
        parsed = parse_command(response_text)
        if parsed is not None:
            return parsed, "model"
    except Exception as exc:  # noqa: BLE001
        return (
            fallback_command(observation),
            f"deterministic_baseline_after_{type(exc).__name__.lower()}",
        )

    return fallback_command(observation), "deterministic_baseline_after_unparsed_output"


def run_task(env: GSTInvoiceGymEnv, client: OpenAI | None, task_id: str) -> TaskRun:
    result = env.reset(task_id=task_id, seed=11)
    observation = result.observation

    print_task_header(task_id, observation.invoice_id, observation.difficulty)

    final_decision: str | None = None
    steps_taken = 0

    for step_index in range(1, MAX_STEPS + 1):
        command, policy = model_command(client, observation)
        action = GSTInvoiceAction(command=command, notes=f"policy={policy}")
        result = env.step(action)
        observation = result.observation
        steps_taken = step_index

        print_step_line(
            step_index=step_index,
            command=command,
            reward=float(result.reward or 0.0),
            grader_score=observation.grader_score,
            task_score=observation.task_score,
            done=result.done,
            policy=policy,
        )

        if result.done:
            final_decision = observation.final_decision
            break

    print_task_footer(observation.task_score, steps_taken, final_decision)

    return TaskRun(
        task_id=task_id,
        score=observation.task_score,
        steps=steps_taken,
        final_decision=final_decision,
    )


def main() -> None:
    llm_client = build_client()
    env = GSTInvoiceGymEnv(base_url=DEFAULT_ENV_BASE_URL).sync()

    runs: list[TaskRun] = []
    with env:
        for task_id in TASKS:
            runs.append(run_task(env, llm_client, task_id))

    average_score = sum(run.score for run in runs) / len(runs)
    print_summary(runs, average_score)
    summary = {
        "average_score": round(average_score, 2),
        "tasks": [
            {
                "task_id": run.task_id,
                "score": round(run.score, 2),
                "steps": run.steps,
                "final_decision": run.final_decision,
            }
            for run in runs
        ],
    }
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
