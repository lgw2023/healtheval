import json
from dataclasses import asdict, dataclass
from datetime import datetime
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
        # 每次运行都使用新的缓存，不加载已有的
        self._data: Dict[str, Any] = {}

    def get(self, key: CacheKey) -> Optional[Any]:
        return self._data.get(key.to_string())

    def set(self, key: CacheKey, value: Any) -> None:
        self._data[key.to_string()] = value

    def flush(self) -> None:
        """持久化当前缓存内容到磁盘。

        约定：
        - ``self.path`` 视作「基础文件名」，真实写入的文件始终带有时间戳后缀；
        - 即便是第一次运行也不会直接写到 ``cache.json``，而是例如
          ``cache_20251202_110307.json``；
        - 若同一秒内被多次调用，则在时间戳相同的情况下继续追加一个数字后缀，
          以避免覆盖（极小概率兜底逻辑）。
        """

        base_path = self.path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = base_path.stem
        suffix = base_path.suffix

        # 优先使用「带时间戳」的文件名；若极端情况下已存在，则追加计数后缀。
        candidate = base_path.parent / f"{stem}_{timestamp}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = base_path.parent / f"{stem}_{timestamp}_{counter}{suffix}"
            counter += 1

        with candidate.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
