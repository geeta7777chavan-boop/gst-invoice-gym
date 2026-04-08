from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow running this file directly from the tests directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import GSTInvoiceAction
from server.app import app
from server.gst_invoice_gym_environment import GSTInvoiceGymEnvironment


def test_direct_environment_easy_task_reaches_full_score() -> None:
    env = GSTInvoiceGymEnvironment()

    observation = env.reset(task_id="easy_invalid_supplier", seed=7)
    assert observation.task_id == "easy_invalid_supplier"
    assert observation.invoice_id == "GST-002"
    assert observation.task_score == 0.0

    observation = env.step(GSTInvoiceAction(command="check_supplier_identity"))
    assert observation.grader_score == 1.0
    assert observation.reward == 0.6
    assert observation.done is False

    observation = env.step(GSTInvoiceAction(command="check_mandatory_fields"))
    assert observation.grader_score == 0.55
    assert observation.done is False

    observation = env.step(GSTInvoiceAction(command="reject"))
    assert observation.done is True
    assert observation.final_decision == "reject"
    assert observation.task_score == 1.0
    assert observation.reward == 1.0


def test_direct_environment_medium_task_scores_in_range() -> None:
    env = GSTInvoiceGymEnvironment()

    observation = env.reset(task_id="medium_tax_regime_mismatch", seed=11)
    assert observation.difficulty == "medium"
    assert observation.steps_remaining == 5

    for command in ("check_tax_regime", "check_tax_math", "reject"):
        observation = env.step(GSTInvoiceAction(command=command))

    assert observation.done is True
    assert observation.task_score == 1.0
    assert 0.0 <= observation.grader_score <= 1.0


def test_direct_environment_rejects_task_case_mismatch() -> None:
    env = GSTInvoiceGymEnvironment()

    with pytest.raises(ValueError, match="not valid for task_id"):
        env.reset(task_id="easy_invalid_supplier", case_id="GST-006")


def test_fastapi_http_endpoints_expose_openenv_contract() -> None:
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"

    metadata_response = client.get("/metadata")
    assert metadata_response.status_code == 200
    metadata = metadata_response.json()
    assert metadata["name"] == "gst_invoice_gym"
    assert "GST invoice" in metadata["description"]

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert {"action", "observation", "state"} <= set(schema)

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks_payload = tasks_response.json()
    assert len(tasks_payload["tasks"]) >= 3
    assert tasks_payload["grader_contract"]["score_range"] == [0.0, 1.0]

    reset_response = client.post(
        "/reset",
        json={"task_id": "medium_tax_regime_mismatch", "seed": 7},
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["observation"]["task_id"] == "medium_tax_regime_mismatch"
    assert reset_payload["done"] is False

    step_response = client.post(
        "/step",
        json={"action": {"command": "check_tax_regime"}},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert 0.0 <= step_payload["reward"] <= 1.0
    assert 0.0 <= step_payload["observation"]["grader_score"] <= 1.0

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert "episode_id" in state_payload
    assert "step_count" in state_payload


def test_websocket_session_preserves_trajectory_state() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {"type": "reset", "data": {"task_id": "hard_manual_review_needed", "seed": 11}}
        )
        reset_message = websocket.receive_json()
        assert reset_message["type"] == "observation"
        assert (
            reset_message["data"]["observation"]["task_id"]
            == "hard_manual_review_needed"
        )

        websocket.send_json({"type": "step", "data": {"command": "check_buyer_identity"}})
        step_message = websocket.receive_json()
        assert step_message["type"] == "observation"
        assert (
            step_message["data"]["observation"]["check_status"]["buyer_identity"]
            == "fail"
        )

        websocket.send_json({"type": "state"})
        state_message = websocket.receive_json()
        assert state_message["type"] == "state"
        assert "buyer_identity" in state_message["data"]["completed_checks"]

        websocket.send_json({"type": "step", "data": {"command": "check_mandatory_fields"}})
        websocket.receive_json()
        websocket.send_json({"type": "step", "data": {"command": "flag_for_review"}})
        done_message = websocket.receive_json()
        assert done_message["data"]["done"] is True
        assert done_message["data"]["observation"]["task_score"] == 1.0
        assert done_message["data"]["observation"]["final_decision"] == "flag"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
