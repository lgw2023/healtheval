import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Sample:
    """Single evaluation sample.

    Attributes:
        sample_id: Identifier from the CSV (row index if not provided).
        query: User question or conversation seed.
        last_answer_phone: Optional previous assistant reply.
        modules_block: Pre-formatted module text block.
        a_answer: Candidate answer A.
        b_answer: Candidate answer B.
        winner: Human annotated winner label ("A" or "B").
        extra: Additional dynamic columns.
    """

    sample_id: str
    query: str
    last_answer_phone: Optional[str]
    modules_block: str
    a_answer: str
    b_answer: str
    winner: str
    extra: dict


class CSVDataLoader:
    """Load evaluation samples from a CSV file.

    The loader keeps extra columns in ``Sample.extra`` so downstream modules can
    use teacher-model context or RAG traces without changing core parsing logic.

    Column names are normalized to support both英文/中文字段。可覆盖 ``column_mapping``
    以匹配新的数据格式。
    """

    DEFAULT_COLUMN_MAPPING = {
        "query": ["query", "典型query", "question"],
        "last_answer_phone": ["last_answer_phone", "last_answer"],
        "modules_block": ["modules_block"],
        "a_answer": ["a_answer", "answer_a"],
        "b_answer": ["b_answer", "answer_b"],
        "winner": ["winner"],
    }

    def __init__(self, path: Path, id_column: str = "id", column_mapping: Optional[dict] = None):
        self.path = Path(path)
        self.id_column = id_column
        self.column_mapping = column_mapping or self.DEFAULT_COLUMN_MAPPING

    def load(self, limit: Optional[int] = None) -> List[Sample]:
        rows: List[Sample] = []
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if limit is not None and len(rows) >= limit:
                    break
                sample_id = row.get(self.id_column) or str(idx)
                sample = Sample(
                    sample_id=sample_id,
                    query=self._get_first(row, "query"),
                    last_answer_phone=self._get_first(row, "last_answer_phone") or None,
                    modules_block=self._get_first(row, "modules_block")
                    or self._compose_modules_block(row),
                    a_answer=self._get_first(row, "a_answer"),
                    b_answer=self._get_first(row, "b_answer"),
                    winner=(self._get_first(row, "winner") or "").strip(),
                    extra={
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            self.id_column,
                            *self.column_mapping.get("query", []),
                            *self.column_mapping.get("last_answer_phone", []),
                            *self.column_mapping.get("modules_block", []),
                            *self.column_mapping.get("a_answer", []),
                            *self.column_mapping.get("b_answer", []),
                            *self.column_mapping.get("winner", []),
                            "data",
                            "suggest",
                            "rag",
                        }
                    },
                )
                rows.append(sample)
        return rows

    @staticmethod
    def _compose_modules_block(row: dict) -> str:
        parts = []
        for key in ("data", "suggest", "rag"):
            if row.get(key):
                parts.append(f"[{key}]\n{row[key]}")
        return "\n\n".join(parts)

    def _get_first(self, row: dict, logical_name: str) -> str:
        candidates = self.column_mapping.get(logical_name, [])
        for name in candidates:
            if name in row and row[name] is not None:
                return row[name]
        return ""


def batched(iterable: Iterable[Sample], batch_size: int) -> Iterable[List[Sample]]:
    """Yield items from *iterable* in batches of *batch_size*.

    This helper is useful for controlling API cost when running evaluations
    in small subsets before scaling to the full dataset.
    """

    batch: List[Sample] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
