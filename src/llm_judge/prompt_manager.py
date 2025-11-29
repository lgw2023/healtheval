from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from . import templates


@dataclass
class PromptVersion:
    name: str
    template: str
    description: str = ""


class PromptManager:
    """Manage and render prompt templates for Ground / Structure judges."""

    def __init__(self):
        self._versions: Dict[str, PromptVersion] = {}

    def register(self, name: str, template: str, description: str = "") -> None:
        self._versions[name] = PromptVersion(name=name, template=template, description=description)

    def get(self, name: str) -> PromptVersion:
        if name not in self._versions:
            raise KeyError(f"Prompt version '{name}' not registered")
        return self._versions[name]

    def render(self, name: str, variables: Mapping[str, str]) -> str:
        version = self.get(name)
        content = version.template
        for key, value in variables.items():
            content = content.replace(f"{{{key}}}", value)
        return content

    @classmethod
    def default_manager(cls) -> "PromptManager":
        manager = cls()
        try:
            from init_prompt import GROUND_PROMPT_TPL, STRUCT_PROMPT_TPL
        except ImportError as exc:
            raise ImportError("init_prompt.py is required to bootstrap default prompts") from exc

        manager.register("ground", GROUND_PROMPT_TPL, "Grounding/consistency judge")
        manager.register("structure", STRUCT_PROMPT_TPL, "Structure/policy judge")
        return manager


def build_prompt_variables(input_data: str, modules_block: str, answer: str) -> dict:
    return {
        "input_data": input_data,
        "modules_block": modules_block,
        "answer": answer,
    }
