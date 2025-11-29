from dataclasses import dataclass
from typing import Dict

from .data_loader import Sample


DIALOGUE_TPL = """用户：{query}
助手：{last_answer_phone}
"""


@dataclass
class InputTemplateResult:
    input_data: str
    modules_block: str


class InputTemplater:
    """Compose conversation-like input and module blocks from a ``Sample``."""

    def __call__(self, sample: Sample) -> InputTemplateResult:
        input_data = sample.query
        if sample.last_answer_phone:
            input_data = DIALOGUE_TPL.format(
                query=sample.query,
                last_answer_phone=sample.last_answer_phone,
            )
        return InputTemplateResult(input_data=input_data, modules_block=sample.modules_block)


class ModuleFormatter:
    """Format RAG/module columns into a text block for prompts."""

    @staticmethod
    def to_block(modules: Dict[str, str]) -> str:
        segments = []
        for name, payload in modules.items():
            if not payload:
                continue
            segments.append(f"## {name}\n{payload}")
        return "\n\n".join(segments)
