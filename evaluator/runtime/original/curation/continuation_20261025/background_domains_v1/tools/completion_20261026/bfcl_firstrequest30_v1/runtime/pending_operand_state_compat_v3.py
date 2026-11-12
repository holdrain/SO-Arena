"""Apply the v4 pending-operand audit to the generic v2 calculator wrapper.

The source implementation is the frozen v4 pending_operand_state_v1 wrapper.
This adapter preserves its same-menu provenance and next-request visibility
checks while accepting the v2 overlay's structured candidate choice values.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    import json
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _selected_candidate_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("kind") == "candidate" and isinstance(value.get("id"), str):
        return value["id"]
    return None


def install(hooks: Any) -> dict[str, Any]:
    cls = hooks.GenericCalculatorBFCLAgent
    if getattr(cls, "_pending_operand_state_v1_installed", False):
        return {"status": "already_installed", "class": cls.__name__}

    original_init = cls.__init__
    original_state = cls._calculator_state
    original_run = cls.run

    def patched_init(self: Any, agent: Any, runner: Any, calculator_module: Any) -> None:
        original_init(self, agent, runner, calculator_module)
        self._pending_selected_left_operand = None
        self._pending_original_choose = agent._choose

        def choose_with_pending_context(
            stage: str,
            turn_index: int,
            state: dict[str, Any],
            instructions: str,
            options: list[dict[str, Any]],
        ) -> dict[str, Any]:
            if stage == "calculator:right_operand":
                current = state.get("current_calculation_in_progress", {})
                visible = current.get("selected_left_operand") if isinstance(current, dict) else None
                expected = self._pending_selected_left_operand
                if expected is not None and visible != expected:
                    raise RuntimeError("right-operand request omitted or changed the pending selected left operand")
                append_jsonl(agent.out / "pending_operand_state_audit.jsonl", {
                    "stage": stage,
                    "operation": current.get("operation") if isinstance(current, dict) else None,
                    "selected_left_operand": copy.deepcopy(visible),
                    "pending_left_operand_visible_before_choice": bool(expected is not None and visible == expected),
                    "source": "the same model-selected candidate and displayed provenance from the live left-operand menu",
                    "model_loaded": bool(getattr(getattr(agent.worker, "ready", {}), "get", lambda *_: False)("model_loaded", False)),
                    "inference_performed": not bool(getattr(agent.worker, "scripted_test", False)),
                })

            selected = self._pending_original_choose(stage, turn_index, state, instructions, options)
            selected_value = selected.get("value") if isinstance(selected, dict) else None
            if stage == "calculator:left_operand":
                candidate_id = _selected_candidate_id(selected_value)
                if candidate_id is not None:
                    chosen_option = next((option for option in options if option.get("value") == selected_value), None)
                    if chosen_option is None:
                        raise RuntimeError("selected calculator candidate is absent from its displayed left-operand menu")
                    self._pending_selected_left_operand = {
                        "candidate_id": candidate_id,
                        "option_description": str(chosen_option.get("description", "")),
                    }
                else:
                    # Back and Stop discard only the uncommitted calculation.
                    self._pending_selected_left_operand = None
            elif stage == "calculator:right_operand":
                # The selected left operand is visible in the request just sent.
                self._pending_selected_left_operand = None
            return selected

        agent._choose = choose_with_pending_context

    def patched_calculator_state(
        self: Any,
        turn_index: int,
        operation: str | None = None,
        *,
        pending: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = original_state(self, turn_index, operation, pending=pending)
        if pending is not None:
            current = state.setdefault("current_calculation_in_progress", {})
            if pending.get("operation") is not None:
                current["operation"] = pending["operation"]
        if operation is not None and self._pending_selected_left_operand is not None:
            state.setdefault("current_calculation_in_progress", {})[
                "selected_left_operand"
            ] = copy.deepcopy(self._pending_selected_left_operand)
        return state

    def patched_run(self: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return original_run(self, **kwargs)
        finally:
            self.agent._choose = self._pending_original_choose

    cls.__init__ = patched_init
    cls._calculator_state = patched_calculator_state
    cls.run = patched_run
    cls._pending_operand_state_v1_installed = True
    return {
        "status": "installed",
        "class": cls.__name__,
        "source_wrapper": "v4 pending_operand_state_v1 semantics, adapted to candidate v2 dict-valued menus",
        "state_field": "current_calculation_in_progress.selected_left_operand",
        "fields": ["candidate_id", "option_description"],
        "scope": "the next calculator right-operand request only; cleared after that choice or cancellation",
    }
