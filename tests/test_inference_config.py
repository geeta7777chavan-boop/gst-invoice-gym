from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from models import GSTInvoiceObservation


def _load_inference_module(monkeypatch, **env: str):
    for key in [
        "API_BASE_URL",
        "MODEL_NAME",
        "OPENAI_MODEL",
        "OPENENV_MODEL",
        "LITELLM_MODEL",
        "API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    sys.modules.pop("inference", None)
    module = importlib.import_module("inference")
    return importlib.reload(module)


def _make_observation() -> GSTInvoiceObservation:
    return GSTInvoiceObservation(
        task_id="easy_invalid_supplier",
        difficulty="easy",
        invoice_id="GST-002",
        scenario="Malformed supplier GSTIN should be rejected.",
        objective="Find the compliance issue and make the correct decision.",
        invoice_features={"invoice_total": 1180.0},
        check_status={
            "supplier_identity": "unknown",
            "buyer_identity": "unknown",
            "tax_regime": "unknown",
            "tax_math": "unknown",
            "mandatory_fields": "unknown",
        },
        recommended_checks=["supplier_identity", "mandatory_fields"],
        available_actions=[
            "check_supplier_identity",
            "check_buyer_identity",
            "check_tax_regime",
            "check_tax_math",
            "check_mandatory_fields",
            "approve",
            "reject",
            "flag_for_review",
        ],
        compliance_issues_found=[],
        last_feedback="",
        grader_score=0.0,
        task_score=0.0,
        final_decision=None,
        steps_remaining=4,
    )


def test_build_client_uses_validator_proxy_credentials(monkeypatch) -> None:
    module = _load_inference_module(
        monkeypatch,
        API_BASE_URL="https://proxy.example/v1",
        API_KEY="validator-key",
        OPENAI_API_KEY="local-key-should-not-win",
        HF_TOKEN="legacy-key-should-not-win",
    )

    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)

    assert module.API_KEY == "validator-key"
    assert module.build_client() is not None
    assert captured == {
        "base_url": "https://proxy.example/v1",
        "api_key": "validator-key",
    }


def test_build_client_accepts_hf_token_as_proxy_key_fallback(monkeypatch) -> None:
    module = _load_inference_module(
        monkeypatch,
        API_BASE_URL="https://proxy.example/v1",
        HF_TOKEN="hf-token-proxy-fallback",
    )

    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)

    assert module.API_KEY == "hf-token-proxy-fallback"
    assert module.build_client() is not None
    assert captured == {
        "base_url": "https://proxy.example/v1",
        "api_key": "hf-token-proxy-fallback",
    }


def test_build_client_ignores_openai_api_key(monkeypatch) -> None:
    module = _load_inference_module(
        monkeypatch,
        API_BASE_URL="https://proxy.example/v1",
        OPENAI_API_KEY="local-key-should-not-be-used",
    )

    assert module.API_KEY is None
    assert module.build_client() is None


def test_model_command_discovers_proxy_model_when_model_name_missing(monkeypatch) -> None:
    module = _load_inference_module(
        monkeypatch,
        API_BASE_URL="https://proxy.example/v1",
        API_KEY="validator-key",
    )

    completion_calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[SimpleNamespace(id="proxy-discovered-model")]
                )
            )
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create_completion)
            )

        def _create_completion(self, **kwargs):
            completion_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"command":"reject","reason":"supplier gstin invalid"}'
                        )
                    )
                ]
            )

    observation = _make_observation()
    command, policy = module.model_command(FakeClient(), observation)

    assert module.MODEL_NAME is None
    assert command == "reject"
    assert policy == "model"
    assert completion_calls[0]["model"] == "proxy-discovered-model"
    assert module.RESOLVED_MODEL_NAME == "proxy-discovered-model"
