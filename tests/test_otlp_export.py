"""OTLP/HTTP export verified against a local mock collector (stands in for Langfuse's /api/public/otel)."""

import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from ragobs.tracing import OTelExporter, langfuse_otlp_config


def test_langfuse_config_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")
    url, headers = langfuse_otlp_config("http://lf:3000/")
    assert url == "http://lf:3000/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode()


def test_no_langfuse_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_otlp_config() is None


def test_spans_reach_otlp_collector(make_traced_pipeline, monkeypatch):
    # sandboxes / corporate networks often set HTTP(S)_PROXY; the local collector must bypass it
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            req = ExportTraceServiceRequest()
            req.ParseFromString(body)
            received.append((self.path, self.headers.get("Authorization"), req))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        pipe, tracer, _, _ = make_traced_pipeline("2 s [px4.md#c000].")
        otlp = OTelExporter(
            "dronedocs-rag",
            otlp_endpoint=f"http://127.0.0.1:{srv.server_port}/api/public/otel/v1/traces",
            otlp_headers={"Authorization": "Basic abc"},
            simple=True,
        )
        tracer.exporters = [otlp]
        pipe.ask("disarm?")
        otlp.flush()
    finally:
        srv.shutdown()
    assert received, "collector got nothing"
    path, auth, _req = received[0]
    assert path == "/api/public/otel/v1/traces" and auth == "Basic abc"
    names = {
        s.name
        for rs in (r[2] for r in received)
        for rs_ in rs.resource_spans
        for ss in rs_.scope_spans
        for s in ss.spans
    }
    assert {"rag.request", "rag.retrieve", "rag.rerank", "rag.generate", "rag.validate_citations"} <= names
