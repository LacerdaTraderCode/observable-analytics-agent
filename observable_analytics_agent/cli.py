"""Command line interface."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from observable_analytics_agent.dataset.generator import DEFAULT_SEED, generate_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("var/analytics.duckdb")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observable_analytics_agent",
        description="A data analyst agent with tracing, cost tracking and an evaluation gate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate-dataset", help="Create the seeded synthetic dataset as a DuckDB file."
    )
    generate.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def run_generate_dataset(path: Path, seed: int) -> int:
    summary = generate_dataset(path, seed)
    logger.info("Dataset written to %s (seed %d)", path, seed)
    for table_name, row_count in summary.row_counts.items():
        logger.info("  %-16s %6d rows", table_name, row_count)
    logger.info("  fingerprint      %s", summary.fingerprint)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = build_parser().parse_args(argv)
    return run_generate_dataset(arguments.path, arguments.seed)
