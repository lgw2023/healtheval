import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float
    top_k: int | None
    top_p: float | None
    prompt_version: str


class SamplingController:
    """Generate run configurations for grid search and repeated sampling."""

    def __init__(self, seeds: Sequence[int] | None = None):
        self.seeds = list(seeds) if seeds else []

    def grid(self, temperatures: Sequence[float], top_ks: Sequence[int | None], top_ps: Sequence[float | None], prompt_versions: Sequence[str]) -> List[DecodeConfig]:
        return [
            DecodeConfig(temp, top_k, top_p, prompt)
            for temp, top_k, top_p, prompt in itertools.product(temperatures, top_ks, top_ps, prompt_versions)
        ]

    def iter_runs(self, configs: Iterable[DecodeConfig], repeats: int) -> Iterator[tuple[int, DecodeConfig]]:
        """Yield ``(seed, config)`` pairs for every repeat.

        Seeds are drawn deterministically from the provided seed list or generated
        randomly to make runs reproducible while supporting arbitrary repeat counts.
        """

        for idx in range(repeats):
            seed = self.seeds[idx % len(self.seeds)] if self.seeds else random.randint(0, 1_000_000)
            for config in configs:
                yield seed, config
