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


def _format_signed_reward(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):+0.2f}"
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
        marker = "Current" if task.task_id == active_task_id else "Ready"
        lines.extend(
            [
                f"**{task.difficulty.title()}** | `{task.task_id}` | {marker}",
                f"Title: {task.title}",
                f"Goal: {task.objective}",
                f"Max steps: `{task.max_steps}` | Case: `{task.default_case_id}`",
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
            f"**Step reward:** `{reward}`",
            f"**Grader score:** `{grader_score}`",
            f"**Task score:** `{task_score}`",
            f"**Episode done:** `{done}` | **Decision:** `{final_decision}`",
            "",
            "**Reward rules**",
            "",
            "- `+0.60` failing recommended check found",
            "- `+0.50` failing non-recommended check found",
            "- `+0.25` recommended check passed",
            "- `+0.10` non-recommended check passed",
            "- `+0.00` duplicate check",
            "- Final decision reward = final task score, capped at `+0.99`",
            "",
            "**Scope**",
            "",
            "- India-only GST benchmark",
            "- Covers GSTIN validation, IGST/CGST/SGST logic, tax math, and mandatory fields",
            "- Does not yet model VAT/GST rules for other countries",
        ]
    )


def _build_scoreboard() -> dict[str, dict[str, str]]:
    return {
        task.task_id: {
            "difficulty": task.difficulty.title(),
            "status": "Ready",
            "latest_score": "-",
            "latest_reward": "-",
            "final_decision": "-",
        }
        for task in TASKS.values()
    }


def _scoreboard_rows(
    scoreboard: dict[str, dict[str, str]],
    active_task_id: str | None,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for task in TASKS.values():
        row = scoreboard.get(task.task_id, {})
        status = row.get("status", "Ready")
        if active_task_id == task.task_id and status == "Ready":
            status = "Selected"
        rows.append(
            [
                row.get("difficulty", task.difficulty.title()),
                task.task_id,
                status,
                row.get("latest_score", "-"),
                row.get("latest_reward", "-"),
                row.get("final_decision", "-"),
            ]
        )
    return rows


def _update_scoreboard_for_reset(
    scoreboard: dict[str, dict[str, str]],
    task_id: str,
) -> dict[str, dict[str, str]]:
    updated = {key: dict(value) for key, value in scoreboard.items()}
    for key, row in updated.items():
        if key == task_id:
            row["status"] = "In progress"
            row["latest_reward"] = "-"
            row["final_decision"] = "-"
        elif row.get("status") == "In progress":
            row["status"] = "Ready"
    return updated


def _update_scoreboard_for_step(
    scoreboard: dict[str, dict[str, str]],
    observation: dict[str, Any],
    reward: Any,
    done: bool,
) -> dict[str, dict[str, str]]:
    updated = {key: dict(value) for key, value in scoreboard.items()}
    task_id = str(observation.get("task_id", ""))
    if task_id in updated:
        row = updated[task_id]
        row["latest_score"] = _format_scalar(observation.get("task_score"))
        row["latest_reward"] = _format_signed_reward(reward)
        row["final_decision"] = observation.get("final_decision") or "-"
        row["status"] = "Completed" if done else "In progress"
    return updated


def _metrics_html(
    reward: str,
    done: str,
    grader_score: str,
    task_score: str,
    final_decision: str,
    steps_remaining: str,
) -> str:
    cards = [
        ("Step Reward", reward),
        ("Done", done),
        ("Grader Score", grader_score),
        ("Task Score", task_score),
        ("Decision", final_decision),
        ("Steps Left", steps_remaining),
    ]
    card_html = "".join(
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;min-width:150px;'>"
        f"<div style='font-size:12px;color:#9da7b3;text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:28px;font-weight:700;margin-top:6px;color:#f0f6fc;'>{value}</div>"
        f"</div>"
        for label, value in cards
    )
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 8px 0;'>"
        f"{card_html}</div>"
    )


def _history_rows(web_manager: Any) -> list[list[str]]:
    logs = getattr(getattr(web_manager, "episode_state", None), "action_logs", []) or []
    if not logs:
        return []

    rows: list[list[str]] = []
    for entry in logs[-10:]:
        action = getattr(entry, "action", {}) or {}
        action_name = action.get("command") or action.get("message") or json.dumps(action)
        reward = _format_signed_reward(getattr(entry, "reward", None))
        done = "yes" if getattr(entry, "done", False) else "no"
        step_count = getattr(entry, "step_count", "?")
        rows.append([str(step_count), str(action_name), reward, done])
    return rows


def _render_dashboard(
    data: dict[str, Any],
    web_manager: Any,
    status_text: str,
    scoreboard: dict[str, dict[str, str]],
) -> tuple[
    str,
    str,
    str,
    list[list[str]],
    list[list[str]],
    str,
    str,
    list[list[str]],
    list[list[str]],
    str,
    str,
    str,
    dict[str, dict[str, str]],
]:
    observation = data.get("observation", {}) or {}
    state = web_manager.get_state()

    reward = _format_signed_reward(data.get("reward"))
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

    metrics_md = _metrics_html(
        reward,
        done,
        grader_score,
        task_score,
        final_decision,
        steps_remaining,
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
    updated_scoreboard = _update_scoreboard_for_step(
        scoreboard,
        observation,
        data.get("reward"),
        bool(data.get("done")),
    )
    performance_rows = _scoreboard_rows(
        updated_scoreboard,
        observation.get("task_id"),
    )
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
        performance_rows,
        history_rows,
        response_json,
        state_json,
        status_text,
        updated_scoreboard,
    )


def _empty_dashboard(
    message: str,
    scoreboard: dict[str, dict[str, str]] | None = None,
    active_task_id: str | None = None,
) -> tuple[
    str,
    str,
    str,
    list[list[str]],
    list[list[str]],
    str,
    str,
    list[list[str]],
    list[list[str]],
    str,
    str,
    str,
    dict[str, dict[str, str]],
]:
    current_scoreboard = scoreboard or _build_scoreboard()
    return (
        "## GST Review Snapshot",
        _metrics_html("-", "no", "-", "-", "pending", "-"),
        _task_catalog_markdown(active_task_id),
        [],
        [],
        "### Guidance",
        _reward_guide_markdown("-", "-", "-", "no", "pending"),
        _scoreboard_rows(current_scoreboard, active_task_id),
        [],
        "",
        "",
        message,
        current_scoreboard,
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
    task_choices = list(TASKS)

    async def reset_dashboard(task_id: str, scoreboard: dict[str, dict[str, str]]):
        selected_task_id = task_id or task_choices[0]
        active_scoreboard = _update_scoreboard_for_reset(
            scoreboard or _build_scoreboard(),
            selected_task_id,
        )
        try:
            data = await web_manager.reset_environment({"task_id": selected_task_id})
            return _render_dashboard(
                data,
                web_manager,
                f"Reset task `{selected_task_id}` successfully.",
                active_scoreboard,
            )
        except Exception as exc:  # pragma: no cover
            return _empty_dashboard(
                f"Error: {exc}",
                scoreboard=active_scoreboard,
                active_task_id=selected_task_id,
            )

    async def step_dashboard(
        task_id: str,
        command: str,
        notes: str,
        scoreboard: dict[str, dict[str, str]],
    ):
        selected_task_id = task_id or task_choices[0]
        current_task_id = web_manager.get_state().get("task_id")
        if not command:
            return _empty_dashboard(
                "Please choose a command before stepping.",
                scoreboard=scoreboard,
                active_task_id=current_task_id or selected_task_id,
            )
        action_data = {"command": command}
        if notes and notes.strip():
            action_data["notes"] = notes.strip()
        try:
            active_scoreboard = scoreboard or _build_scoreboard()
            status_prefix = ""
            if current_task_id != selected_task_id:
                active_scoreboard = _update_scoreboard_for_reset(
                    active_scoreboard,
                    selected_task_id,
                )
                await web_manager.reset_environment({"task_id": selected_task_id})
                status_prefix = f"Auto-reset to `{selected_task_id}`. "
            data = await web_manager.step_environment(action_data)
            return _render_dashboard(
                data,
                web_manager,
                f"{status_prefix}Executed `{command}` successfully.",
                active_scoreboard,
            )
        except Exception as exc:  # pragma: no cover
            return _empty_dashboard(
                f"Error: {exc}",
                scoreboard=scoreboard,
                active_task_id=current_task_id or selected_task_id,
            )

    def state_only():
        try:
            state_json = json.dumps(web_manager.get_state(), indent=2)
            return state_json, "Fetched current environment state."
        except Exception as exc:  # pragma: no cover
            return "", f"Error: {exc}"

    with gr.Blocks(title="GST Invoice Dashboard") as demo:
        scoreboard_state = gr.State(value=_build_scoreboard())
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ## GST Invoice Dashboard

                    Pick a task, reset the environment, and inspect the GST review
                    with cleaner metrics and an all-task performance board.
                    """
                )
                task_selector = gr.Dropdown(
                    choices=task_choices,
                    label="Task",
                    value=task_choices[0] if task_choices else None,
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
                    reset_btn = gr.Button("Reset")
                    step_btn = gr.Button("Step", variant="primary")
                    state_btn = gr.Button("Get State")
                status = gr.Textbox(label="Status", interactive=False)
                performance_board = gr.Dataframe(
                    headers=[
                        "Difficulty",
                        "Task",
                        "Status",
                        "Latest score",
                        "Latest reward",
                        "Decision",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="All task performance",
                    value=_scoreboard_rows(_build_scoreboard(), None),
                )
                history = gr.Dataframe(
                    headers=["Step", "Action", "Reward", "Done"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Current task history",
                    value=[],
                )

            with gr.Column(scale=2):
                overview = gr.Markdown(value="## GST Review Snapshot")
                metrics = gr.HTML(
                    value=_metrics_html("-", "no", "-", "-", "pending", "-")
                )
                reward_guide = gr.Markdown(
                    value=_reward_guide_markdown("-", "-", "-", "no", "pending")
                )
                guidance = gr.Markdown(value="### Guidance")
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
                with gr.Accordion("Debug JSON", open=False):
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
            inputs=[task_selector, scoreboard_state],
            outputs=[
                overview,
                metrics,
                task_catalog,
                invoice_features,
                check_status,
                guidance,
                reward_guide,
                performance_board,
                history,
                response_json,
                state_json,
                status,
                scoreboard_state,
            ],
        )
        step_btn.click(
            fn=step_dashboard,
            inputs=[task_selector, command, notes, scoreboard_state],
            outputs=[
                overview,
                metrics,
                task_catalog,
                invoice_features,
                check_status,
                guidance,
                reward_guide,
                performance_board,
                history,
                response_json,
                state_json,
                status,
                scoreboard_state,
            ],
        )
        state_btn.click(
            fn=state_only,
            outputs=[state_json, status],
        )

    return demo
