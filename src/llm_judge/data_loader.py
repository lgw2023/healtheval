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
    """

    def __init__(self, path: Path, id_column: str = "id"):
        self.path = Path(path)
        self.id_column = id_column

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
                    query=row.get("query", ""),
                    last_answer_phone=row.get("last_answer_phone") or None,
                    modules_block=row.get("modules_block")
                    or self._compose_modules_block(row),
                    a_answer=row.get("a_answer", ""),
                    b_answer=row.get("b_answer", ""),
                    winner=(row.get("winner") or "").strip(),
                    extra={k: v for k, v in row.items() if k not in {
                        self.id_column,
                        "query",
                        "last_answer_phone",
                        "modules_block",
                        "a_answer",
                        "b_answer",
                        "winner",
                    }},
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
