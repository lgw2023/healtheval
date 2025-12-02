from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedLog:
    """结构化后的日志解析结果，方便后续渲染 HTML。"""

    config: Optional[Dict[str, Any]]
    eval_metrics: Dict[str, float]
    extra_numbers: Dict[str, float]
    human_counts: Dict[str, int]
    llm_counts: Dict[str, int]


def _read_lines(log_path: Path) -> List[str]:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # 去掉终端颜色控制符
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return text.splitlines()


def _parse_pipeline_config(lines: List[str]) -> Optional[Dict[str, Any]]:
    """解析缓存日志开头的 PIPELINE CONFIG JSON。"""
    start = None
    for i, line in enumerate(lines):
        if "========== PIPELINE CONFIG ==========" in line:
            start = i
            break
    if start is None:
        return None

    json_lines: List[str] = []
    brace_level = 0
    in_json = False
    for line in lines[start + 1 :]:
        if "{" in line:
            in_json = True
        if not in_json:
            continue
        json_lines.append(line.strip())
        brace_level += line.count("{")
        brace_level -= line.count("}")
        if brace_level == 0 and "}" in line:
            break

    raw = "\n".join(json_lines)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_eval_summary(lines: List[str]) -> ParsedLog:
    """从日志尾部解析最终指标与胜负分布。"""
    eval_metrics: Dict[str, float] = {}
    extra_numbers: Dict[str, float] = {}
    human_counts: Dict[str, int] = {}
    llm_counts: Dict[str, int] = {}

    # 解析 EVALUATION SUMMARY 区块
    summary_start = None
    for i, line in enumerate(lines):
        if "========== EVALUATION SUMMARY ==========" in line:
            summary_start = i
            break
    if summary_start is not None:
        for line in lines[summary_start + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("----"):
                continue

            # 简单的 key=value 形式（alpha / pair_acc / tie_rate / chi_square / alt_test_p）
            m = re.match(r"([a-zA-Z_]+)=([0-9.+-eE]+)", stripped)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    eval_metrics[key] = float(val)
                except ValueError:
                    pass
                continue

            # samples_decisions=9 (answers=18)
            if stripped.startswith("samples_decisions="):
                m2 = re.search(r"samples_decisions=(\d+)", stripped)
                if m2:
                    extra_numbers["samples_decisions"] = float(m2.group(1))
                m3 = re.search(r"answers=(\d+)", stripped)
                if m3:
                    extra_numbers["answers"] = float(m3.group(1))
                continue

            # human_winner_dist: A=6, B=3
            if stripped.startswith("human_winner_dist"):
                for k, v in re.findall(r"([AB])=(\d+)", stripped):
                    human_counts[k] = int(v)
                continue

            # llm_winner_dist:   A=6, B=3
            if stripped.startswith("llm_winner_dist"):
                for k, v in re.findall(r"([AB])=(\d+)", stripped):
                    llm_counts[k] = int(v)
                continue

    # 解析 Metrics 中的 DE/DO 与 Alt-Test 人工 & LLM 计数（作为兜底或补充）
    for line in lines:
        if "[Metrics] 所有打分的全局均值 mean=" in line:
            m = re.search(r"mean=([0-9.+-eE]+)", line)
            if m:
                extra_numbers.setdefault("global_mean", float(m.group(1)))
        elif "[Metrics] 期望误差平方和 DE=" in line:
            m = re.search(r"DE=([0-9.+-eE]+)", line)
            if m:
                extra_numbers.setdefault("DE", float(m.group(1)))
        elif "总体观测误差平方和 DO=" in line:
            m = re.search(r"DO=([0-9.+-eE]+)", line)
            if m:
                extra_numbers.setdefault("DO", float(m.group(1)))
        elif "Alt-Test：总体数据一致性" in line:
            # 后面会有 human_counts / llm_counts 详细信息
            continue
        elif "human_counts={" in line and "human胜负计数" not in line:
            # [Metrics] 人工胜负计数 human_counts={'A': 6, 'B': 3}
            inner = re.search(r"human_counts=({.*})", line)
            if inner:
                try:
                    d = eval(inner.group(1), {"__builtins__": {}})  # 安全起见禁用内建
                    for k, v in d.items():
                        if isinstance(v, int):
                            human_counts.setdefault(k, v)
                except Exception:
                    pass
        elif "llm_counts={" in line:
            inner = re.search(r"llm_counts=({.*})", line)
            if inner:
                try:
                    d = eval(inner.group(1), {"__builtins__": {}})
                    for k, v in d.items():
                        if isinstance(v, int):
                            llm_counts.setdefault(k, v)
                except Exception:
                    pass

    # 再补一次 Report 段落里的 alpha/pair-accuracy 等（如果没在 summary 中拿到）
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[Report]  单条打分稳定性"):
            m = re.search(r"α=([0-9.+-eE]+)", stripped)
            if m and "alpha" not in eval_metrics:
                eval_metrics["alpha"] = float(m.group(1))
        elif "pair-accuracy=" in stripped and "pair_acc" not in eval_metrics:
            m1 = re.search(r"pair-accuracy=([0-9.+-eE]+)", stripped)
            m2 = re.search(r"tie_rate=([0-9.+-eE]+)", stripped)
            if m1:
                eval_metrics["pair_acc"] = float(m1.group(1))
            if m2:
                eval_metrics["tie_rate"] = float(m2.group(1))
        elif "Alt-Test: chi_square=" in stripped and "chi_square" not in eval_metrics:
            m1 = re.search(r"chi_square=([0-9.+-eE]+)", stripped)
            m2 = re.search(r"p_value=([0-9.+-eE]+)", stripped)
            if m1:
                eval_metrics["chi_square"] = float(m1.group(1))
            if m2:
                eval_metrics["alt_test_p"] = float(m2.group(1))

    return ParsedLog(
        config=None,  # 由外层单独填充
        eval_metrics=eval_metrics,
        extra_numbers=extra_numbers,
        human_counts=human_counts,
        llm_counts=llm_counts,
    )


def parse_log(log_path: Path) -> ParsedLog:
    lines = _read_lines(log_path)
    cfg = _parse_pipeline_config(lines)
    parsed = _parse_eval_summary(lines)
    parsed.config = cfg
    return parsed


def _render_metric_bars(eval_metrics: Dict[str, float]) -> str:
    """把 0~1 区间内的指标画成简单的水平条形图。"""
    bars_html: List[str] = []
    # 约定这些指标在 [0, 1] 区间，chi_square 单独展示数值
    bar_keys = ["alpha", "pair_acc", "tie_rate", "alt_test_p"]

    for key in bar_keys:
        if key not in eval_metrics:
            continue
        val = eval_metrics[key]
        # 截断到 [0, 1] 区间
        width = max(0.0, min(1.0, float(val))) * 100.0
        label = f"{key} = {val:.4f}"
        bars_html.append(
            f"""
            <div class="metric-row">
              <div class="metric-label">{html.escape(label)}</div>
              <div class="bar-bg">
                <div class="bar-fill" style="width: {width:.1f}%;"></div>
              </div>
            </div>
            """
        )

    # chi_square 如果存在，就仅数值展示
    if "chi_square" in eval_metrics:
        val = eval_metrics["chi_square"]
        bars_html.append(
            f"""
            <div class="metric-row metric-text-only">
              <div class="metric-label">chi_square = {val:.4f}</div>
            </div>
            """
        )
    return "\n".join(bars_html)


def _render_counts_section(human_counts: Dict[str, int], llm_counts: Dict[str, int]) -> str:
    all_labels = sorted(set(human_counts.keys()) | set(llm_counts.keys()))
    if not all_labels:
        return "<p>没有解析到胜负分布数据。</p>"

    max_count = max(
        [c for c in human_counts.values()] + [c for c in llm_counts.values()] + [1]
    )

    rows: List[str] = []
    for label in all_labels:
        h = human_counts.get(label, 0)
        l = llm_counts.get(label, 0)
        h_width = h / max_count * 100.0
        l_width = l / max_count * 100.0
        rows.append(
            f"""
            <div class="count-row">
              <div class="count-label">类别 {html.escape(label)}</div>
              <div class="count-bars">
                <div class="count-group">
                  <span class="count-title">Human: {h}</span>
                  <div class="bar-bg small">
                    <div class="bar-fill human" style="width: {h_width:.1f}%;"></div>
                  </div>
                </div>
                <div class="count-group">
                  <span class="count-title">LLM: {l}</span>
                  <div class="bar-bg small">
                    <div class="bar-fill llm" style="width: {l_width:.1f}%;"></div>
                  </div>
                </div>
              </div>
            </div>
            """
        )
    return "\n".join(rows)


def build_html(parsed: ParsedLog, raw_log: str) -> str:
    """根据解析结果生成完整 HTML 页面。"""
    cfg = parsed.config or {}

    config_items = []
    for key in [
        "data_path",
        "cache_path",
        "prompt_versions",
        "decode_configs",
        "repeats",
        "limit",
        "seeds",
        "combine_weights",
        "verbose",
    ]:
        if key not in cfg:
            continue
        value = cfg[key]
        pretty = json.dumps(value, ensure_ascii=False) if not isinstance(value, (str, int, float)) else str(value)
        config_items.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(pretty)}</td></tr>"
        )
    config_table = "\n".join(config_items) if config_items else "<tr><td colspan='2'>未解析到配置 JSON。</td></tr>"

    metrics_html = _render_metric_bars(parsed.eval_metrics)
    counts_html = _render_counts_section(parsed.human_counts, parsed.llm_counts)

    extra_items: List[str] = []
    for k, v in parsed.extra_numbers.items():
        # 对浮点数统一保留 4 位小数，其他类型直接转成字符串
        if isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        extra_items.append(
            f"<li><span class='extra-key'>{html.escape(k)}</span>"
            f"<span class='extra-val'>{html.escape(v_str)}</span></li>"
        )
    extra_html = "\n".join(extra_items) if extra_items else "<li>无额外统计数值。</li>"

    # 页面主体
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>HealthEval 日志可视化报告</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      font-weight: 600;
      color: #f9fafb;
    }}
    h1 {{ font-size: 28px; margin-bottom: 20px; }}
    h2 {{ font-size: 22px; margin-top: 24px; }}
    h3 {{ font-size: 18px; margin-top: 16px; }}

    .section {{
      margin-bottom: 28px;
      padding: 18px 20px;
      border-radius: 12px;
      background: radial-gradient(circle at top left, rgba(96,165,250,0.18), rgba(15,23,42,0.95));
      border: 1px solid rgba(148,163,184,0.35);
      box-shadow: 0 18px 45px rgba(15,23,42,0.9);
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 4px;
    }}
    .section-subtitle {{
      font-size: 13px;
      color: #9ca3af;
    }}

    table.config-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    table.config-table th,
    table.config-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid rgba(148,163,184,0.35);
      vertical-align: top;
    }}
    table.config-table th {{
      width: 160px;
      color: #cbd5f5;
      text-align: left;
      white-space: nowrap;
    }}
    table.config-table tr:last-child th,
    table.config-table tr:last-child td {{
      border-bottom: none;
    }}

    .metric-row {{
      margin: 8px 0;
    }}
    .metric-label {{
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .bar-bg {{
      position: relative;
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: rgba(30,64,175,0.45);
      overflow: hidden;
    }}
    .bar-bg.small {{
      height: 8px;
    }}
    .bar-fill {{
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      border-radius: 999px;
      background: linear-gradient(90deg, #38bdf8, #a855f7);
    }}
    .bar-fill.human {{
      background: linear-gradient(90deg, #22c55e, #a3e635);
    }}
    .bar-fill.llm {{
      background: linear-gradient(90deg, #f97316, #facc15);
    }}

    .metric-text-only .bar-bg {{
      display: none;
    }}

    .count-row {{
      margin: 10px 0 14px;
      padding-bottom: 6px;
      border-bottom: 1px dashed rgba(148,163,184,0.35);
    }}
    .count-label {{
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 4px;
    }}
    .count-bars {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 20px;
      align-items: center;
    }}
    .count-group {{
      min-width: 200px;
    }}
    .count-title {{
      font-size: 12px;
      color: #cbd5f5;
      display: inline-block;
      margin-bottom: 2px;
    }}

    .extra-metrics {{
      list-style: none;
      padding-left: 0;
      margin: 6px 0 0;
      font-size: 13px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 6px 14px;
    }}
    .extra-metrics li {{
      color: #e5e7eb;
    }}
    .extra-key {{
      color: #9ca3af;
      margin-right: 4px;
    }}

    details.raw-log {{
      margin-top: 6px;
      font-size: 12px;
    }}
    details.raw-log summary {{
      cursor: pointer;
      color: #93c5fd;
    }}
    pre.log-text {{
      max-height: 420px;
      overflow: auto;
      margin-top: 8px;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(15,23,42,0.95);
      border: 1px solid rgba(30,64,175,0.7);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 11px;
      line-height: 1.35;
      white-space: pre;
    }}

    .tag-row {{
      margin-bottom: 12px;
      font-size: 13px;
      color: #9ca3af;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(30,64,175,0.4);
      border: 1px solid rgba(59,130,246,0.7);
      margin-right: 6px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="section">
      <div class="section-header">
        <h1>HealthEval 评估日志可视化</h1>
        <span class="section-subtitle">基于 cache.log 自动解析与呈现</span>
      </div>
      <div class="tag-row">
        <span class="pill">模块分段展示</span>
        <span class="pill">标题 / 内容分级</span>
        <span class="pill">数值图形化</span>
      </div>
      <p style="font-size:13px;color:#d1d5db;margin:0;">
        本页面由 <code>log_visualizer.py</code> 自动生成，用于快速理解一次评估运行的配置与指标结果。
      </p>
    </header>

    <section class="section">
      <div class="section-header">
        <h2>一、运行配置概览</h2>
        <span class="section-subtitle">Pipeline Config（从日志 JSON 自动解析）</span>
      </div>
      <table class="config-table">
        <tbody>
          {config_table}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>二、核心指标总览</h2>
        <span class="section-subtitle">Krippendorff’s α、pair-accuracy、Alt-Test 等</span>
      </div>
      <h3>2.1 区间型指标（0~1）可视化</h3>
      {metrics_html}

      <h3>2.2 其他统计数值</h3>
      <ul class="extra-metrics">
        {extra_html}
      </ul>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>三、人类标注 vs LLM 胜负分布</h2>
        <span class="section-subtitle">A / B 胜场对比</span>
      </div>
      {counts_html}
    </section>

    <section class="section">
      <div class="section-header">
        <h2>四、原始日志（可折叠查看）</h2>
        <span class="section-subtitle">作为结构化结果的补充信息</span>
      </div>
      <details class="raw-log">
        <summary>展开查看完整 cache.log 文本</summary>
        <pre class="log-text">{html.escape(raw_log)}</pre>
      </details>
    </section>
  </div>
</body>
</html>
"""
    return html_doc


def generate_html(
    log_path: Path,
    output_path: Path,
) -> Path:
    """主入口：从日志文件生成 HTML 报告。"""
    lines = _read_lines(log_path)
    parsed = parse_log(log_path)
    raw_log = "\n".join(lines)
    html_content = build_html(parsed, raw_log)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解析 HealthEval 的 cache.log，并生成可视化 HTML 报告。",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="日志文件路径，默认使用项目根目录下的 cache/cache.log",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 HTML 路径，默认写入 cache/report.html",
    )
    args = parser.parse_args()

    # 推断项目根目录：当前文件在 src/llm_judge/ 下
    project_root = Path(__file__).resolve().parents[2]
    default_log = project_root / "cache" / "cache.log"
    default_out = project_root / "cache" / "report.html"

    log_path = Path(args.log) if args.log else default_log
    out_path = Path(args.out) if args.out else default_out

    generated = generate_html(log_path, out_path)
    print(f"[log_visualizer] HTML 报告已生成: {generated}")


if __name__ == "__main__":
    main()


