from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT / "src"))
except:
    sys.path.insert(0, str("/Users/liguowei/ubuntu/healtheval/src"))


from llm_judge.llm_client import build_llm_caller
from llm_judge.pipeline import EvaluationPipeline
from llm_judge.prompt_manager import PromptManager
from llm_judge.reporting import ReportBuilder
from llm_judge.sampling import DecodeConfig
from llm_judge.logging_utils import setup_cache_and_logging
from llm_judge.log_visualizer import generate_html as generate_log_html


def _validate_llm_env(model_config: str | None = None) -> None:
    """Ensure required environment variables for LLM calling exist.

    The evaluation should stop early with a clear message when credentials are
    missing to avoid partially executed runs or unclear stack traces.
    """

    if model_config and model_config.lower() != "default":
        suffix = model_config.upper()
        required = [
            f"LLM_MODEL_{suffix}_URL",
            f"LLM_MODEL_{suffix}_NAME",
            f"LLM_MODEL_{suffix}_API_KEY",
        ]
    else:
        required = ["LLM_MODEL_URL", "LLM_MODEL_NAME", "LLM_MODEL_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            "Missing required environment variables for LLM access "
            f"(model_config={model_config or 'default'}): {missing_list}. "
            "Set them or rerun with --mock to use the offline scorer."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    parser.add_argument("--data", type=Path, default=Path("data.csv"), help="Path to CSV data file")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache") / "cache.json",
        help="Path to JSON cache file (will be stored under the 'cache' directory by default)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated scores per (query, answer) for stability (Krippendorff's alpha)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Top-k sampling value")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Top-p sampling value")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["ground", "structure"],
        help="Prompt versions to evaluate (default: ground structure)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM caller instead of real endpoint (for offline debugging)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample / per-round progress and metrics.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发调用大模型的工作线程数（>1 时启用并发，默认 4）。",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="default",
        help=(
            "选择要使用的模型配置名称："
            "默认使用 LLM_MODEL_URL/LLM_MODEL_NAME/LLM_MODEL_API_KEY；"
            "若指定为 foo，则使用 "
            "LLM_MODEL_FOO_URL/LLM_MODEL_FOO_NAME/LLM_MODEL_FOO_API_KEY。"
        ),
    )
    parser.add_argument(
        "--on-missing-dims",
        choices=["fill_max", "ignore", "retry"],
        default="retry",
        help=(
            "当模型输出中缺失提示词定义的某些打分维度时的处理策略："
            "fill_max=按默认满分补齐；ignore=只使用模型实际输出的维度；"
            "retry=视为失败并尽量重新让模型打分（仅对本轮 LLM 调用生效，缓存命中不重试）。"
        ),
    )
    return parser.parse_args()


def default_args() -> argparse.Namespace:
    """在交互式环境中使用的默认参数集合。

    示例（在 Python 解释器 / IPython 中）::

        from run_eval import main, default_args
        main(default_args())
    """

    return argparse.Namespace(
        data=Path("data.csv"),
        cache=Path("cache") / "cache.json",
        limit=1,
        repeats=3,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        prompts=["ground", "structure"],
        mock=False,
        verbose=False,
        workers=4,
        on_missing_dims="fill_max",
        model_config="default",
    )


def main(args: argparse.Namespace | None = None) -> None:
    # 优先从 .env 等文件加载环境变量，便于本地开发与多环境切换
    load_dotenv()

    # 脚本模式：从命令行解析；交互模式：可以显式传入 default_args()
    if args is None:
        args = parse_args()
        # args = default_args()

    # 统一规范 cache 路径，并根据 cache 文件路径设置对应的日志文件 tee。
    # 为了便于区分不同「真实模型」的运行结果，这里优先从环境变量中读取当前使用的模型名，
    # 作为文件名后缀；不直接使用 model_config 字符串。
    if getattr(args, "cache", None) is not None:
        model_name_suffix: str | None = None
        if not args.mock:
            model_config = getattr(args, "model_config", None)
            if model_config and model_config.lower() != "default":
                suffix = model_config.upper()
                model_key = f"LLM_MODEL_{suffix}_NAME"
            else:
                model_key = "LLM_MODEL_NAME"
            model_name_suffix = os.getenv(model_key)

        cache_path, log_path = setup_cache_and_logging(
            args.cache,
            name_suffix=model_name_suffix,
        )
        args.cache = cache_path
    else:
        log_path = None
    if not args.mock:
        _validate_llm_env(getattr(args, "model_config", None))
    caller = build_llm_caller(
        mock=args.mock,
        model_config=getattr(args, "model_config", None),
    )
    prompt_manager = PromptManager.default_manager()
    configs = [
        DecodeConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            prompt_version=prompt,
        )
        for prompt in args.prompts
    ]

    seeds = [42]
    combine_weights = {"ground": 2.0, "structure": 1.0}

    pipeline = EvaluationPipeline(
        data_path=args.data,
        caller=caller,
        cache_path=args.cache,
        prompt_manager=prompt_manager,
        verbose=args.verbose,
        max_workers=args.workers,
        missing_dims_strategy=getattr(args, "on_missing_dims", "fill_max"),
        missing_full_score=5.0,
    )

    # 结构化打印本次运行的 EvaluationPipeline 配置信息，便于检查与复现
    pipeline_info = pipeline.describe(
        configs=configs,
        repeats=args.repeats,
        limit=args.limit,
        seeds=seeds,
        combine_weights=combine_weights,
    )
    print("\n========== PIPELINE CONFIG ==========")
    print(json.dumps(pipeline_info, ensure_ascii=False, indent=2))

    reports = pipeline.run(
        configs=configs,
        repeats=args.repeats,
        limit=args.limit,
        seeds=seeds,
        combine_weights=combine_weights,
    )

    print("\n========== EVALUATION SUMMARY ==========")
    for config, report in reports.items():
        print("---- 配置 ----")
        if isinstance(config, str):
            print(f"Prompt={config} | temp={args.temperature} | top_k={args.top_k} | top_p={args.top_p}")
        else:
            print(
                f"Prompt={config.prompt_version} | temp={config.temperature} | "
                f"top_k={config.top_k} | top_p={config.top_p}"
            )
        print("---- 指标 ----")
        # 使用 ReportBuilder 的 pretty 文本输出，展示更多统计信息
        print(ReportBuilder.to_pretty_text(report))
        print("----------------------------------------\n")

    # 评估完成后，尝试基于本次日志文件生成可视化 HTML 报告
    try:
        if log_path is not None:
            # 日志文件与 cache 位于同一目录，HTML 报告沿用相同的文件名 stem，后缀改为 .html
            html_path = log_path.with_suffix(".html")
            generate_log_html(log_path, html_path)
            print(f"[run_eval] 日志可视化报告已生成: {html_path}")
    except Exception as e:
        # 可视化失败不影响主评估流程，仅打印提示
        print(f"[run_eval] 生成日志可视化报告时出错: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

