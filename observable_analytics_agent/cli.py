"""Command line interface."""

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from observable_analytics_agent.dataset.generator import DEFAULT_SEED, generate_dataset
from observable_analytics_agent.evaluation.baseline import (
    DEFAULT_BASELINE_PATH,
    BaselineError,
    baseline_from_metrics,
    load_baseline,
    save_baseline,
)
from observable_analytics_agent.evaluation.gate import RERECORD_COMMAND, evaluate_gate
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.evaluation.metrics import summarize
from observable_analytics_agent.evaluation.render import render_evaluation, report_payload
from observable_analytics_agent.evaluation.runner import (
    DEFAULT_DATASET_PATH,
    EvaluationSetup,
    prepare_evaluation,
)
from observable_analytics_agent.evaluation.tasks import GoldenTask
from observable_analytics_agent.models.anthropic_model import DEFAULT_MODEL, MissingApiKeyError
from observable_analytics_agent.models.recording import DEFAULT_RECORDINGS_DIR, ModelMode
from observable_analytics_agent.observability.ledger import (
    DEFAULT_LEDGER_PATH,
    LedgerNotFoundError,
    RunLedger,
)
from observable_analytics_agent.observability.report import build_report, render_report
from observable_analytics_agent.prompts.registry import (
    DEFAULT_PROMPTS_DIR,
    MAIN_PROMPT_NAME,
    PromptError,
    PromptRegistry,
    hash_content,
)

logger = logging.getLogger(__name__)

HASH_PREFIX_CHARS = 12
USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observable_analytics_agent",
        description="A data analyst agent with tracing, cost tracking and an evaluation gate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_generate_dataset(commands)
    _add_report(commands)
    _add_prompts(commands)
    _add_evaluate(commands)
    return parser


def _add_generate_dataset(commands: argparse._SubParsersAction) -> None:
    generate = commands.add_parser(
        "generate-dataset", help="Create the seeded synthetic dataset as a DuckDB file."
    )
    generate.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.set_defaults(handler=_generate_dataset)


def _add_report(commands: argparse._SubParsersAction) -> None:
    report = commands.add_parser(
        "report",
        help="Print cost, latency, success rate and tool error rate from the run ledger.",
    )
    report.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    report.set_defaults(handler=_report)


def _add_prompts(commands: argparse._SubParsersAction) -> None:
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


def _add_evaluate(commands: argparse._SubParsersAction) -> None:
    evaluate = commands.add_parser(
        "evaluate",
        help="Run the golden tasks and score them. With --gate, compare with the baseline.",
    )
    evaluate.add_argument(
        "--mode",
        choices=[mode.value for mode in ModelMode],
        default=ModelMode.REPLAY.value,
        help="replay reads committed recordings, record calls the API and stores the replies, "
        "live calls the API without storing.",
    )
    evaluate.add_argument("--prompt", default=MAIN_PROMPT_NAME)
    evaluate.add_argument("--prompt-version", help="Defaults to the latest version.")
    evaluate.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    evaluate.add_argument(
        "--task",
        action="append",
        dest="task_ids",
        help="Run only this task. Repeat for several. A subset cannot be gated.",
    )
    evaluate.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    evaluate.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    evaluate.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    evaluate.add_argument(
        "--gate", action="store_true", help="Exit with 1 when the baseline or a budget is violated."
    )
    evaluate.add_argument(
        "--write-baseline",
        action="store_true",
        help="Store this run's metrics as the new baseline.",
    )
    evaluate.add_argument("--output", type=Path, help="Write the full report as JSON.")
    evaluate.set_defaults(handler=_evaluate)


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


def _select_tasks(task_ids: list[str] | None) -> list[GoldenTask] | None:
    if not task_ids:
        return list(GOLDEN_TASKS)
    by_id = {task.task_id: task for task in GOLDEN_TASKS}
    unknown = [task_id for task_id in task_ids if task_id not in by_id]
    if unknown:
        logger.error("Unknown task(s): %s. Known: %s", ", ".join(unknown), ", ".join(by_id))
        return None
    return [by_id[task_id] for task_id in task_ids]


def _evaluate(arguments: argparse.Namespace) -> int:
    if arguments.gate and arguments.write_baseline:
        logger.error("--gate and --write-baseline cannot be combined: a new baseline passes itself.")
        return USAGE_ERROR
    if arguments.task_ids and (arguments.gate or arguments.write_baseline):
        logger.error("A subset of tasks cannot be gated or stored as a baseline.")
        return USAGE_ERROR
    tasks = _select_tasks(arguments.task_ids)
    if tasks is None:
        return USAGE_ERROR
    setup = EvaluationSetup(
        mode=ModelMode(arguments.mode),
        model=arguments.model,
        prompt_name=arguments.prompt,
        prompt_version=arguments.prompt_version,
        recordings_dir=arguments.recordings_dir,
        dataset_path=arguments.dataset,
        ledger_path=arguments.ledger,
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    expected_baseline = None
    if arguments.gate:
        try:
            expected_baseline = load_baseline(arguments.baseline)
        except BaselineError as error:
            logger.error("%s Record one with `%s`.", error, RERECORD_COMMAND)
            return 1
    try:
        prepared = prepare_evaluation(setup, os.environ)
    except (MissingApiKeyError, PromptError) as error:
        logger.error("%s", error)
        return 1
    results = prepared.harness.run(tasks)
    metrics = summarize(results)
    if arguments.write_baseline:
        baseline = baseline_from_metrics(
            metrics, prepared.prompt, prepared.model, datetime.now(UTC).date()
        )
        save_baseline(baseline, arguments.baseline)
        logger.info("Baseline written to %s", arguments.baseline)
    gate = None
    if expected_baseline is not None:
        gate = evaluate_gate(
            metrics,
            expected_baseline,
            prompt_version=prepared.prompt.version,
            model=prepared.model,
        )
    sys.stdout.write(render_evaluation(results, metrics, gate))
    if arguments.output:
        payload = report_payload(
            results,
            metrics,
            gate,
            prompt=prepared.prompt,
            model=prepared.model,
            mode=prepared.mode,
        )
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if gate is None or gate.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)
