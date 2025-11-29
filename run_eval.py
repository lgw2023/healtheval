from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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
    parser.add_argument("--cache", type=Path, default=None, help="Path to JSON cache file")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per config")
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


def main() -> None:
    # 优先从 .env 等文件加载环境变量，便于本地开发与多环境切换
    load_dotenv()

    args = parse_args()
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

    pipeline = EvaluationPipeline(
        data_path=args.data,
        caller=caller,
        cache_path=args.cache,
        prompt_manager=prompt_manager,
        verbose=args.verbose,
    )
    reports = pipeline.run(configs=configs, repeats=args.repeats, limit=args.limit, seeds=[42])

    print("\n========== EVALUATION SUMMARY ==========")
    for config, report in reports.items():
        print("---- 配置 ----")
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

