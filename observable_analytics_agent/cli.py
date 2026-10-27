"""Command line interface."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from observable_analytics_agent.dataset.generator import DEFAULT_SEED, generate_dataset
from observable_analytics_agent.observability.ledger import (
    DEFAULT_LEDGER_PATH,
    LedgerNotFoundError,
    RunLedger,
)
from observable_analytics_agent.observability.report import build_report, render_report
from observable_analytics_agent.prompts.registry import (
    DEFAULT_PROMPTS_DIR,
    PromptError,
    PromptRegistry,
    hash_content,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("var/analytics.duckdb")
HASH_PREFIX_CHARS = 12


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
    generate.set_defaults(handler=_generate_dataset)

    report = commands.add_parser(
        "report",
        help="Print cost, latency, success rate and tool error rate from the run ledger.",
    )
    report.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    report.set_defaults(handler=_report)

    prompts = commands.add_parser("prompts", help="Inspect and verify the versioned prompts.")
    prompt_commands = prompts.add_subparsers(dest="prompts_command", required=True)
    for name, help_text, handler in (
        ("list", "List every prompt version with its hash and changelog.", _prompts_list),
        ("verify", "Check every prompt file against the hash in the manifest.", _prompts_verify),
    ):
        command = prompt_commands.add_parser(name, help=help_text)
        command.add_argument("--directory", type=Path, default=DEFAULT_PROMPTS_DIR)
        command.set_defaults(handler=handler)
    hash_command = prompt_commands.add_parser(
        "hash", help="Print the hash to declare in the manifest for a prompt file."
    )
    hash_command.add_argument("path", type=Path)
    hash_command.set_defaults(handler=_prompts_hash)
    return parser


def _generate_dataset(arguments: argparse.Namespace) -> int:
    summary = generate_dataset(arguments.path, arguments.seed)
    logger.info("Dataset written to %s (seed %d)", arguments.path, arguments.seed)
    for table_name, row_count in summary.row_counts.items():
        logger.info("  %-16s %6d rows", table_name, row_count)
    logger.info("  fingerprint      %s", summary.fingerprint)
    return 0


def _report(arguments: argparse.Namespace) -> int:
    try:
        report = build_report(RunLedger(arguments.ledger))
    except LedgerNotFoundError as error:
        logger.error("%s", error)
        return 1
    sys.stdout.write(render_report(report))
    return 0


def _load_registry(directory: Path) -> PromptRegistry | None:
    try:
        return PromptRegistry.load(directory)
    except PromptError as error:
        logger.error("%s", error)
        return None


def _prompts_list(arguments: argparse.Namespace) -> int:
    registry = _load_registry(arguments.directory)
    if registry is None:
        return 1
    for prompt in registry.all():
        short_hash = prompt.content_hash[:HASH_PREFIX_CHARS]
        sys.stdout.write(f"{prompt.name} {prompt.version}  {short_hash}  {prompt.changelog}\n")
    return 0


def _prompts_verify(arguments: argparse.Namespace) -> int:
    registry = _load_registry(arguments.directory)
    if registry is None:
        return 1
    sys.stdout.write(f"{len(registry.all())} prompt versions verified.\n")
    return 0


def _prompts_hash(arguments: argparse.Namespace) -> int:
    sys.stdout.write(hash_content(arguments.path.read_bytes()) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)
