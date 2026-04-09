---
title: GST Invoice OpenEnv
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
base_path: /web
short_description: OpenEnv GST invoice compliance environment.
tags:
  - openenv
  - reinforcement-learning
  - fastapi
  - docker
  - gst
  - invoice-compliance
---

# GST Invoice OpenEnv Environment

This repository packages a real-world GST invoice processing environment for OpenEnv. The agent reviews structured invoice data, runs targeted compliance checks, and decides whether to `approve`, `reject`, or `flag_for_review` a document. The environment is fully synthetic, deterministic, and self-contained, so it is reproducible inside Docker and suitable for automated evaluation.

## Live Space

- Hugging Face Space: [Geeta1980/gst-invoice-gym](https://huggingface.co/spaces/Geeta1980/gst-invoice-gym)
- Runtime URL: [geeta1980-gst-invoice-gym.hf.space](https://geeta1980-gst-invoice-gym.hf.space)

The deployed Space includes both the default OpenEnv playground and a GST-specific custom dashboard, so reviewers can inspect and interact with the environment visually without relying only on raw JSON.

## Why this environment is useful

GST invoice review is a real operational workflow in finance and compliance teams. A useful agent must inspect multiple fields, detect tax and identity problems, avoid unnecessary checks, and make the right downstream decision. This environment turns that workflow into a standardized RL benchmark with typed OpenEnv models, deterministic graders, and shaped rewards.

## OpenEnv compliance

The project follows the OpenEnv simulation pattern:

- Typed Pydantic `Action`, `Observation`, and `State` models in [models.py](/D:/openenv-invoicegym/models.py)
- Stateful environment implementing `reset()`, `step()`, and `state` in [server/gst_invoice_gym_environment.py](/D:/openenv-invoicegym/server/gst_invoice_gym_environment.py)
- OpenEnv FastAPI app in [server/app.py](/D:/openenv-invoicegym/server/app.py)
- Reviewer-friendly custom web dashboard in [server/gst_invoice_dashboard.py](/D:/openenv-invoicegym/server/gst_invoice_dashboard.py)
- Root metadata manifest in [openenv.yaml](/D:/openenv-invoicegym/openenv.yaml)
- Root baseline script in [inference.py](/D:/openenv-invoicegym/inference.py)
- Docker support in [Dockerfile](/D:/openenv-invoicegym/Dockerfile) and [server/Dockerfile](/D:/openenv-invoicegym/server/Dockerfile)

## Tasks

The environment exposes 3 deterministic tasks with easy, medium, and hard difficulty progression:

| Task ID | Difficulty | Goal | Expected correct decision |
| --- | --- | --- | --- |
| `easy_invalid_supplier` | easy | Detect a malformed supplier GSTIN and avoid over-checking | `reject` |
| `medium_tax_regime_mismatch` | medium | Detect wrong interstate tax regime plus related math inconsistency | `reject` |
| `hard_manual_review_needed` | hard | Detect a buyer-identity anomaly that should be escalated, not auto-rejected | `flag` |

Task definitions and synthetic cases live in [task_definitions.py](/D:/openenv-invoicegym/task_definitions.py) and [data/invoices.json](/D:/openenv-invoicegym/data/invoices.json).

Each task is intentionally bound to a specific deterministic invoice case so the task objective, visible scenario text, and grader ground truth cannot drift apart.

## Action space

The agent submits a typed `GSTInvoiceAction` with a `command` field. Valid commands are:

- `check_supplier_identity`
- `check_buyer_identity`
- `check_tax_regime`
- `check_tax_math`
- `check_mandatory_fields`
- `approve`
- `reject`
- `flag_for_review`

The first five commands are inspection steps. The last three are terminal decisions.

## Observation space

Each `GSTInvoiceObservation` contains:

- `task_id`, `difficulty`, `invoice_id`, `scenario`, `objective`
- `invoice_features`: structured visible invoice attributes
- `check_status`: per-check status, one of `unknown`, `pass`, or `fail`
- `recommended_checks`: compliance checks the grader expects the agent to prioritize
- `available_actions`
- `compliance_issues_found`
- `last_feedback`
- `grader_score`: deterministic score for the latest action in the range `0.0` to `1.0`
- `task_score`: current normalized task score in the range `0.0` to `1.0`
- `final_decision`
- `steps_remaining`

The `state()` endpoint exposes `GSTInvoiceState`, which includes episode-level fields such as `completed_checks`, `detected_issue_checks`, `total_reward`, and `step_count`.

## Graders

The environment uses deterministic internal graders with explicit score ranges:

1. Check grader
   Scores each inspection action based on whether it surfaced a real issue and whether that check was recommended for the task.
2. Coverage grader
   Measures how much of the recommended evidence-gathering path the agent completed.
3. Decision grader
   Scores the final decision against the task ground truth.

The normalized final task score is:

```text
0.60 * final_decision_accuracy
+ 0.25 * recommended_check_coverage
+ 0.15 * issue_detection
```

All grader-facing scores are clipped to the `0.0` to `1.0` range.

## Reward logic

The environment provides shaped step rewards plus a terminal reward:

- `0.60` for finding a failing recommended check
- `0.50` for finding a failing non-recommended check
- `0.25` for completing a recommended check that passes
- `0.10` for completing a non-recommended check that passes
- `0.00` for repeating a check
- Terminal reward equals the normalized final task score in the range `0.0` to `1.0`
- Timeout episodes receive only a small capped partial credit signal

This gives the agent meaningful feedback over the full trajectory instead of only at episode end.

## Project layout

- [models.py](/D:/openenv-invoicegym/models.py): OpenEnv action, observation, and state models
- [client.py](/D:/openenv-invoicegym/client.py): OpenEnv client wrapper
- [task_definitions.py](/D:/openenv-invoicegym/task_definitions.py): task catalog, case loading, graders, score formulas
- [server/gst_invoice_gym_environment.py](/D:/openenv-invoicegym/server/gst_invoice_gym_environment.py): environment logic
- [server/app.py](/D:/openenv-invoicegym/server/app.py): FastAPI/OpenEnv server
- [tests/test_environment.py](/D:/openenv-invoicegym/tests/test_environment.py): local tests
- [inference.py](/D:/openenv-invoicegym/inference.py): baseline script

## Local setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

The editable install exposes the `gst_invoice_gym` import path used by the Hugging Face Space quick start and the packaged Python client.

Run tests:

```powershell
pytest -q
```

Start the server:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the local environment:

```powershell
openenv validate .
```

Validate a running server:

```powershell
openenv validate --url http://localhost:8000
```

## API surface

OpenEnv routes:

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `GET /state`
- `POST /reset`
- `POST /step`
- `POST /mcp`
- `WebSocket /ws`

Project-specific helper routes:

- `GET /tasks`
- `GET /grader-spec`

## Baseline inference

The required baseline script is [inference.py](/D:/openenv-invoicegym/inference.py). It uses the OpenAI client only through the injected LiteLLM proxy and reads these validator-facing environment variables:

- `API_BASE_URL`
- `API_KEY`

Compatibility fallback:

- `HF_TOKEN` is accepted as a proxy-key alias when `API_KEY` is absent, but the script still sends requests only to `API_BASE_URL`.

Optional model hint variables:

- `MODEL_NAME`
- `OPENAI_MODEL`
- `OPENENV_MODEL`
- `LITELLM_MODEL`

If no model hint is provided, the script first asks the proxy for available models and then falls back to common LiteLLM-compatible model ids. This keeps the baseline on the provided proxy instead of calling OpenAI, Groq, or Hugging Face directly.

Optional local server override:

- `OPENENV_BASE_URL` default: `http://localhost:8000`

Run it after starting the server:

```powershell
python inference.py
```

If the proxy variables are unavailable, the script falls back to a deterministic heuristic policy so the environment can still be smoke-tested locally.

### Reproducible baseline scores

Using the built-in deterministic fallback policy with `seed=11`, the expected scores are:

| Task ID | Score |
| --- | --- |
| `easy_invalid_supplier` | `1.00` |
| `medium_tax_regime_mismatch` | `1.00` |
| `hard_manual_review_needed` | `1.00` |
| Average | `1.00` |

## Docker

Build the root container:

```powershell
docker build -t gst-invoice-gym .
```

Run it:

```powershell
docker run --rm -p 8000:8000 gst-invoice-gym
```

## Hugging Face Space

The repo is structured for OpenEnv-style space deployment. After validating locally:

```powershell
openenv push --repo-id <your-username>/gst-invoice-gym
```

Tag the Space with `openenv` and ensure the deployed URL responds to `/health`, `/metadata`, and `/reset`.

## Validation Status

This environment has been checked across the full intended submission path:

- local `pytest -q`
- local `openenv validate .`
- local Docker run plus `/health`
- deployed Hugging Face Space health check
- deployed `openenv validate --url ...`
- deployed `python inference.py` smoke test
