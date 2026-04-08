from __future__ import annotations

from typing import Any

from task_definitions import ALL_CHECKS, AVAILABLE_COMMANDS, task_catalog

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the GST invoice environment. Install project "
        "dependencies before starting the server."
    ) from exc

try:
    from ..models import GSTInvoiceAction, GSTInvoiceObservation
    from .gst_invoice_dashboard import build_gst_dashboard
    from .gst_invoice_gym_environment import GSTInvoiceGymEnvironment
except ImportError:
    from models import GSTInvoiceAction, GSTInvoiceObservation
    from server.gst_invoice_dashboard import build_gst_dashboard
    from server.gst_invoice_gym_environment import GSTInvoiceGymEnvironment


app = create_app(
    GSTInvoiceGymEnvironment,
    GSTInvoiceAction,
    GSTInvoiceObservation,
    env_name="gst_invoice_gym",
    max_concurrent_envs=4,
    gradio_builder=build_gst_dashboard,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "GST Invoice OpenEnv server is running."}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": task_catalog(),
        "grader_contract": {
            "score_range": [0.0, 1.0],
            "weights": {
                "final_decision_accuracy": 0.60,
                "recommended_check_coverage": 0.25,
                "issue_detection": 0.15,
            },
        },
        "reward_contract": {
            "score_range": [0.0, 1.0],
            "inspection_rewards": {
                "recommended_failure_found": 0.60,
                "non_recommended_failure_found": 0.50,
                "recommended_check_pass": 0.25,
                "non_recommended_check_pass": 0.10,
                "duplicate_check": 0.00,
            },
            "terminal_reward": "reward equals final normalized task score",
        },
    }


@app.get("/grader-spec")
def grader_spec() -> dict[str, Any]:
    return {
        "observation_fields": {
            "check_status": "Per-check pass/fail/unknown status",
            "grader_score": "Latest action score between 0.0 and 1.0",
            "task_score": "Current normalized task score between 0.0 and 1.0",
            "final_decision": "approve, reject, or flag once chosen",
        },
        "action_space": AVAILABLE_COMMANDS,
        "check_space": list(ALL_CHECKS),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
