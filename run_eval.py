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


def _validate_llm_env() -> None:
    """Ensure required environment variables for LLM calling exist.

    The evaluation should stop early with a clear message when credentials are
    missing to avoid partially executed runs or unclear stack traces.
    """

    required = ["LLM_MODEL_URL", "LLM_MODEL_NAME", "LLM_MODEL_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            f"Missing required environment variables for LLM access: {missing_list}. "
            "Set them or rerun with --mock to use the offline scorer."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    parser.add_argument("--data", type=Path, default=Path("data.csv"), help="Path to CSV data file")
    parser.add_argument("--cache", type=Path, default=Path("cache.json"), help="Path to JSON cache file")
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
    return parser.parse_args()


def default_args() -> argparse.Namespace:
    """在交互式环境中使用的默认参数集合。

    示例（在 Python 解释器 / IPython 中）::

        from run_eval import main, default_args
        main(default_args())
    """

    return argparse.Namespace(
        data=Path("data.csv"),
        cache="/Users/liguowei/ubuntu/healtheval/cache.json",
        limit=1,
        repeats=3,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        prompts=["ground", "structure"],
        mock=False,
        verbose=False,
    )


def main(args: argparse.Namespace | None = None) -> None:
    # 优先从 .env 等文件加载环境变量，便于本地开发与多环境切换
    load_dotenv()

    # 脚本模式：从命令行解析；交互模式：可以显式传入 default_args()
    if args is None:
        args = parse_args()
        # args = default_args()
    if not args.mock:
        _validate_llm_env()
    caller = build_llm_caller(mock=args.mock)
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


if __name__ == "__main__":
    main()

