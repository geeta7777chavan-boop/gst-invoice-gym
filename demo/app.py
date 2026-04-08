from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this demo directly from the demo directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.invoice_gym import GSTInvoiceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick GST invoice RL episode.")
    parser.add_argument("--case-id", default="GST-001", help="Synthetic invoice case id.")
    parser.add_argument(
        "--actions",
        nargs="*",
        type=int,
        default=[2, 3, 5],
        help="Discrete actions to execute in order.",
    )
    args = parser.parse_args()

    env = GSTInvoiceEnv()
    observation, info = env.reset(options={"case_id": args.case_id})
    print(json.dumps({"observation": _serialize(observation), "info": info}, indent=2))

    for action in args.actions:
        observation, reward, terminated, truncated, step_info = env.step(action)
        print(
            json.dumps(
                {
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "observation": _serialize(observation),
                    "info": step_info,
                },
                indent=2,
            )
        )
        if terminated or truncated:
            break


def _serialize(observation: dict[str, object]) -> dict[str, object]:
    return {
        "invoice_features": observation["invoice_features"].tolist(),
        "check_status": observation["check_status"].tolist(),
        "remaining_steps": int(observation["remaining_steps"]),
    }


if __name__ == "__main__":
    main()
