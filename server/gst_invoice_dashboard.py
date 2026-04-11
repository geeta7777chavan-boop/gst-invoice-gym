from __future__ import annotations

import json
from typing import Any

import gradio as gr

try:
    from ..task_definitions import TASKS
except ImportError:
    from task_definitions import TASKS


def _labelize(value: str) -> str:
    return value.replace("_", " ").title()


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _mapping_rows(mapping: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in mapping.items():
        if isinstance(value, list):
            pretty_value = ", ".join(str(item) for item in value) or "-"
        elif isinstance(value, dict):
            pretty_value = json.dumps(value)
        else:
            pretty_value = _format_scalar(value)
        rows.append([_labelize(key), pretty_value])
    return rows


def _list_markdown(title: str, values: list[Any]) -> str:
    rendered = ", ".join(f"`{value}`" for value in values) if values else "_None_"
    return f"**{title}:** {rendered}"


def _bullet_block(title: str, values: list[Any]) -> str:
    if not values:
        return f"**{title}**\n\n- _None_"
    bullet_lines = "\n".join(f"- `{value}`" for value in values)
    return f"**{title}**\n\n{bullet_lines}"


def _task_catalog_markdown(active_task_id: str | None) -> str:
    lines = ["## Task Ladder", ""]
    for task in TASKS.values():
        marker = ">>" if task.task_id == active_task_id else "-"
        lines.extend(
            [
                f"{marker} **{task.difficulty.title()}**: `{task.task_id}`",
                f"Title: {task.title}",
                f"Goal: {task.objective}",
                f"Max steps: `{task.max_steps}` | Default case: `{task.default_case_id}`",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _reward_guide_markdown(
    reward: str,
    grader_score: str,
    task_score: str,
    done: str,
    final_decision: str,
) -> str:
    return "\n".join(
        [
            "## Reward Guide",
            "",
            f"**Current step reward:** `{reward}`",
            f"**Current grader score:** `{grader_score}`",
            f"**Current task score:** `{task_score}`",
            f"**Episode done:** `{done}` | **Final decision:** `{final_decision}`",
            "",
            "**How rewards work**",
            "",
            "- `0.60` for finding a failing recommended check",
            "- `0.50` for finding a failing non-recommended check",
            "- `0.25` for completing a recommended check that passes",
            "- `0.10` for completing a non-recommended check that passes",
            "- `0.00` for repeating a check",
            "- Terminal reward equals the final task score, capped at `0.99`",
            "",
            "**Current benchmark scope**",
            "",
            "- India-only GST benchmark",
            "- Covers Indian GST identity, tax-regime, tax-math, and mandatory-field checks",
            "- Does not yet model VAT/GST rules for other countries",
        ]
    )


def _history_rows(web_manager: Any) -> list[list[str]]:
    logs = getattr(getattr(web_manager, "episode_state", None), "action_logs", []) or []
    if not logs:
        return []

    rows: list[list[str]] = []
    for entry in logs[-10:]:
        action = getattr(entry, "action", {}) or {}
        action_name = action.get("command") or action.get("message") or json.dumps(action)
        reward = _format_scalar(getattr(entry, "reward", None))
        done = "yes" if getattr(entry, "done", False) else "no"
        step_count = getattr(entry, "step_count", "?")
        rows.append([str(step_count), str(action_name), reward, done])
    return rows


def _render_dashboard(data: dict[str, Any], web_manager: Any, status_text: str) -> tuple[str, str, str, list[list[str]], list[list[str]], str, list[list[str]], str, str, str]:
    observation = data.get("observation", {}) or {}
    state = web_manager.get_state()

    reward = _format_scalar(data.get("reward"))
    done = "yes" if data.get("done") else "no"
    grader_score = _format_scalar(observation.get("grader_score"))
    task_score = _format_scalar(observation.get("task_score"))
    final_decision = observation.get("final_decision") or "pending"
    steps_remaining = _format_scalar(observation.get("steps_remaining"))

    overview_md = "\n".join(
        [
            "## GST Review Snapshot",
            f"**Task:** `{observation.get('task_id', '-')}`",
            f"**Difficulty:** `{observation.get('difficulty', '-')}`",
            f"**Invoice:** `{observation.get('invoice_id', '-')}`",
            "",
            f"**Scenario:** {observation.get('scenario', '-')}",
            "",
            f"**Objective:** {observation.get('objective', '-')}",
            "",
            f"**Last feedback:** {observation.get('last_feedback', '-')}",
        ]
    )

    metrics_md = "\n".join(
        [
            "### Metrics",
            "",
            "| Reward | Done | Grader score | Task score | Final decision | Steps remaining |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| `{reward}` | `{done}` | `{grader_score}` | `{task_score}` | `{final_decision}` | `{steps_remaining}` |",
        ]
    )
    task_catalog_md = _task_catalog_markdown(observation.get("task_id"))
    reward_guide_md = _reward_guide_markdown(
        reward=reward,
        grader_score=grader_score,
        task_score=task_score,
        done=done,
        final_decision=final_decision,
    )

    available_actions = observation.get("available_actions", []) or []
    inspection_actions = [
        action for action in available_actions if str(action).startswith("check_")
    ]
    decision_actions = [
        action for action in available_actions if not str(action).startswith("check_")
    ]
    guidance_md = "\n\n".join(
        [
            "### Guidance",
            _bullet_block(
                "Recommended checks",
                observation.get("recommended_checks", []) or [],
            ),
            _bullet_block(
                "Detected issues",
                observation.get("compliance_issues_found", []) or [],
            ),
            _bullet_block("Inspection actions", inspection_actions),
            _bullet_block("Decision actions", decision_actions),
        ]
    )

    invoice_rows = _mapping_rows(observation.get("invoice_features", {}) or {})
    check_rows = _mapping_rows(observation.get("check_status", {}) or {})
    history_rows = _history_rows(web_manager)
    response_json = json.dumps(data, indent=2)
    state_json = json.dumps(state, indent=2)

    return (
        overview_md,
        metrics_md,
        task_catalog_md,
        invoice_rows,
        check_rows,
        guidance_md,
        reward_guide_md,
        history_rows,
        response_json,
        state_json,
        status_text,
    )


def _empty_dashboard(message: str) -> tuple[str, str, str, list[list[str]], list[list[str]], str, list[list[str]], str, str, str]:
    return (
        "## GST Review Snapshot",
        "### Metrics",
        _task_catalog_markdown(None),
        [],
        [],
        "### Guidance",
        _reward_guide_markdown("-", "-", "-", "no", "pending"),
        [],
        "",
        "",
        message,
    )


def build_gst_dashboard(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str | None,
) -> gr.Blocks:
    del metadata, is_chat_env, title

    command_choices: list[str] = []
    notes_placeholder = "Optional rationale for the chosen action."
    for field in action_fields:
        if field.get("name") == "command":
            command_choices = list(field.get("choices") or [])
        elif field.get("name") == "notes":
            notes_placeholder = field.get("placeholder") or notes_placeholder

    async def reset_dashboard():
        try:
            data = await web_manager.reset_environment()
            return _render_dashboard(data, web_manager, "Environment reset successfully.")
        except Exception as exc:  # pragma: no cover
            return _empty_dashboard(f"Error: {exc}")

    async def step_dashboard(command: str, notes: str):
        if not command:
            return _empty_dashboard("Please choose a command before stepping.")
        action_data = {"command": command}
        if notes and notes.strip():
            action_data["notes"] = notes.strip()
        try:
            data = await web_manager.step_environment(action_data)
            return _render_dashboard(
                data,
                web_manager,
                f"Executed `{command}` successfully.",
            )
        except Exception as exc:  # pragma: no cover
            return _empty_dashboard(f"Error: {exc}")

    def state_only():
        try:
            state_json = json.dumps(web_manager.get_state(), indent=2)
            return state_json, "Fetched current environment state."
        except Exception as exc:  # pragma: no cover
            return "", f"Error: {exc}"

    with gr.Blocks(title="GST Invoice Dashboard") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ## GST Invoice Dashboard

                    A reviewer-friendly view of the same OpenEnv environment.
                    Use this tab to inspect the task, visible invoice fields,
                    check status progression, and action history without digging
                    through the raw JSON every time.
                    """
                )
                task_catalog = gr.Markdown(value=_task_catalog_markdown(None))
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=False):
                        gr.Markdown(quick_start_md)
                command = gr.Dropdown(
                    choices=command_choices,
                    label="Command",
                    value=command_choices[0] if command_choices else None,
                )
                notes = gr.Textbox(
                    label="Notes",
                    placeholder=notes_placeholder,
                    lines=3,
                )
                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    reset_btn = gr.Button("Reset")
                    state_btn = gr.Button("Get State")
                status = gr.Textbox(label="Status", interactive=False)
                history = gr.Dataframe(
                    headers=["Step", "Action", "Reward", "Done"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Action history",
                    value=[],
                )

            with gr.Column(scale=2):
                overview = gr.Markdown(value="## GST Review Snapshot")
                metrics = gr.Markdown(value="### Metrics")
                guidance = gr.Markdown(value="### Guidance")
                reward_guide = gr.Markdown(
                    value=_reward_guide_markdown("-", "-", "-", "no", "pending")
                )
                with gr.Row():
                    invoice_features = gr.Dataframe(
                        headers=["Invoice field", "Value"],
                        datatype=["str", "str"],
                        interactive=False,
                        wrap=True,
                        label="Visible invoice fields",
                    )
                    check_status = gr.Dataframe(
                        headers=["Compliance check", "Status"],
                        datatype=["str", "str"],
                        interactive=False,
                        wrap=True,
                        label="Current check status",
                    )
                with gr.Row():
                    response_json = gr.Code(
                        label="Latest response",
                        language="json",
                        interactive=False,
                    )
                    state_json = gr.Code(
                        label="Current state",
                        language="json",
                        interactive=False,
                    )

        reset_btn.click(
            fn=reset_dashboard,
            outputs=[
                overview,
                metrics,
                task_catalog,
                invoice_features,
                check_status,
                guidance,
                reward_guide,
                history,
                response_json,
                state_json,
                status,
            ],
        )
        step_btn.click(
            fn=step_dashboard,
            inputs=[command, notes],
            outputs=[
                overview,
                metrics,
                task_catalog,
                invoice_features,
                check_status,
                guidance,
                reward_guide,
                history,
                response_json,
                state_json,
                status,
            ],
        )
        state_btn.click(
            fn=state_only,
            outputs=[state_json, status],
        )

    return demo
