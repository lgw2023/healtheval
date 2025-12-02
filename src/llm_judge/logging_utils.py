from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable, Optional, Tuple


class TeeIO:
    """简单的 tee 对象：同时写入多个底层流。

    用于在终端输出的同时，将所有日志内容同步写入到指定的日志文件中。
    """

    def __init__(self, *streams: IO[str]):
        # 过滤掉 None，避免在某些极端环境下 sys.__stdout__ / __stderr__ 为 None 的情况
        self._streams: list[IO[str]] = [s for s in streams if s is not None]

    def write(self, data: str) -> int:  # type: ignore[override]
        for s in self._streams:
            s.write(data)
        for s in self._streams:
            s.flush()
        # 返回写入长度，以兼容 file-like 协议
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        for s in self._streams:
            s.flush()


def _make_safe_suffix(name: str | None) -> str | None:
    """将任意字符串转换为适合作为文件名后缀的安全形式。"""
    if not name:
        return None
    # 简单归一化：去掉首尾空白，替换空格和常见不安全字符
    safe = name.strip()
    for ch in (" ", "/", "\\", ":", "|", "*", "?", '"', "<", ">", "'"):
        safe = safe.replace(ch, "_")
    return safe or None


def setup_cache_and_logging(
    cache: Path | str | None,
    name_suffix: str | None = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """根据 cache 路径规范化目录结构，并将 stdout/stderr tee 到对应的日志文件。

    约定：
    - cache 文件路径由调用方决定，例如 ``cache/run1_cache.json``；
    - 若提供 ``name_suffix``（例如模型配置名 / 模型名称），
      则会在缓存文件名的 stem 后追加 ``_<safe_suffix>``，
      例如 ``cache.json`` + ``deepinfra`` -> ``cache_deepinfra.json``；
    - 日志文件与 cache 文件位于同一目录下，文件名相同，仅后缀改为 ``.log``；
    - 当同名日志文件已存在时，会在日志文件名后追加时间戳，避免覆盖历史记录。

    返回值：
    - (规范化后的 cache Path（或 None）, 实际写入的 log Path（或 None）)
    """

    if cache is None:
        return None, None

    cache_path = Path(cache)
    safe_suffix = _make_safe_suffix(name_suffix)
    if safe_suffix:
        cache_path = cache_path.with_name(f"{cache_path.stem}_{safe_suffix}{cache_path.suffix}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 日志文件始终带时间戳后缀（即便是第一次运行也不覆盖同名 cache.log）
    base_log = cache_path.with_suffix(".log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_log.parent / f"{base_log.stem}_{timestamp}{base_log.suffix}"

    # 极端情况下（同一秒多次启动）再追加数字后缀，避免覆盖
    counter = 1
    while log_path.exists():
        log_path = base_log.parent / f"{base_log.stem}_{timestamp}_{counter}{base_log.suffix}"
        counter += 1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    # 将 stdout / stderr tee 到日志文件：保证所有 print 的内容都会同时写入 log
    sys.stdout = TeeIO(sys.__stdout__, log_file)  # type: ignore[assignment]
    sys.stderr = TeeIO(sys.__stderr__, log_file)  # type: ignore[assignment]

    return cache_path, log_path


