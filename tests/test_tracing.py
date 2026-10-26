"""The tracer provider prints spans by default and exports over OTLP when configured."""

import io
import json

import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from observable_analytics_agent.observability import tracing


def test_spans_are_printed_to_the_given_stream_by_default() -> None:
    stream = io.StringIO()
    provider = tracing.build_tracer_provider(console_stream=stream)

    with provider.get_tracer("test").start_as_current_span("hello"):
        pass

    printed = json.loads(stream.getvalue())
    assert printed["name"] == "hello"
    assert printed["resource"]["attributes"]["service.name"] == tracing.SERVICE_NAME


def test_an_endpoint_switches_to_otlp_with_the_traces_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints: list[str] = []

    class RecordingExporter(SpanExporter):
        def __init__(self, endpoint: str) -> None:
            endpoints.append(endpoint)

        def export(self, spans) -> SpanExportResult:
            return SpanExportResult.SUCCESS

    monkeypatch.setattr(tracing, "OTLPSpanExporter", RecordingExporter)
    console = io.StringIO()

    provider = tracing.build_tracer_provider("http://collector:4318/", console_stream=console)
    with provider.get_tracer("test").start_as_current_span("hello"):
        pass
    provider.shutdown()

    assert endpoints == ["http://collector:4318/v1/traces"]
    assert console.getvalue() == ""
