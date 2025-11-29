import json
from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheKey:
    prompt_version: str
    temperature: float
    top_k: int | None
    top_p: float | None
    query_id: str
    answer_id: str
    run_idx: int

    def to_string(self) -> str:
        payload = f"{self.prompt_version}|{self.temperature}|{self.top_k}|{self.top_p}|{self.query_id}|{self.answer_id}|{self.run_idx}"
        return md5(payload.encode("utf-8")).hexdigest()


class JSONCache:
    """Filesystem cache for raw LLM responses."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self._data: Dict[str, Any] = json.load(f)
        else:
            self._data = {}

    def get(self, key: CacheKey) -> Optional[Any]:
        return self._data.get(key.to_string())

    def set(self, key: CacheKey, value: Any) -> None:
        self._data[key.to_string()] = value

    def flush(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
