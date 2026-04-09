from __future__ import annotations

import importlib
import sys


def _load_inference_module(monkeypatch, **env: str):
    for key in ["API_BASE_URL", "MODEL_NAME", "API_KEY", "OPENAI_API_KEY", "HF_TOKEN"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    sys.modules.pop("inference", None)
    module = importlib.import_module("inference")
    return importlib.reload(module)


def test_build_client_prefers_validator_api_key(monkeypatch) -> None:
    module = _load_inference_module(
        monkeypatch,
        API_BASE_URL="https://proxy.example/v1",
        MODEL_NAME="proxy-model",
        API_KEY="validator-key",
        OPENAI_API_KEY="local-key-should-not-win",
        HF_TOKEN="legacy-key-should-not-win",
    )

    assert module.API_KEY == "validator-key"
    assert module.build_client() is not None
