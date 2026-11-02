"""Tracer provider setup: console by default, OTLP over HTTP when an endpoint is configured."""

import sys
from typing import TextIO

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

SERVICE_NAME = "observable-analytics-agent"
OTLP_TRACES_PATH = "/v1/traces"


def build_tracer_provider(
    otlp_endpoint: str | None = None,
    console_stream: TextIO | None = None,
    *,
    export_to_console: bool = True,
) -> TracerProvider:
    """Export to an OTLP collector when `otlp_endpoint` is set, otherwise print spans.

    The console exporter writes to stderr by default so that answers stay alone on stdout. An
    evaluation produces hundreds of spans, so it asks for `export_to_console=False`.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint.rstrip("/") + OTLP_TRACES_PATH)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    elif export_to_console:
        console = ConsoleSpanExporter(out=console_stream or sys.stderr)
        provider.add_span_processor(SimpleSpanProcessor(console))
    return provider
