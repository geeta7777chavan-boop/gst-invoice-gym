from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState
except ImportError:
    from models import GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState


class GSTInvoiceGymEnv(
    EnvClient[GSTInvoiceAction, GSTInvoiceObservation, GSTInvoiceState]
):
    """OpenEnv client for the GST invoice compliance environment."""

    def _step_payload(self, action: GSTInvoiceAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[GSTInvoiceObservation]:
        obs_data = payload.get("observation", {})
        observation = GSTInvoiceObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> GSTInvoiceState:
        return GSTInvoiceState.model_validate(payload)
