"""Command line interface: ``dronedocs ingest | ask | eval | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="dronedocs", description="DroneDocs production RAG")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="build the hybrid index from data/raw")
    p_ing.add_argument("--raw-dir")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true")
    p_ev = sub.add_parser("eval", help="run the golden-set evaluation")
    p_ev.add_argument("--limit", type=int)
    p_ev.add_argument("--no-ragas", action="store_true")
    p_ev.add_argument("--no-gate", action="store_true", help="do not exit non-zero below thresholds")
    p_ev.add_argument("--out")
    p_ev.add_argument("--resume", action="store_true", help="reuse finished rows from the last interrupted run")
    p_ev.add_argument("--report-only", action="store_true", help="rebuild the report from the checkpoint, no calls")
    p_ev.add_argument(
        "--offline",
        action="store_true",
        help="deterministic CI gate: replay recorded answers, no LLM/judge/network (uses eval.offline_thresholds)",
    )
    p_srv = sub.add_parser("serve", help="run the FastAPI server")
    p_srv.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container entrypoint
    p_srv.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "ingest":
        from .ingest import ingest

        print(json.dumps(ingest(args.raw_dir), indent=2))
        return 0
    if args.cmd == "ask":
        from .pipeline import RAGPipeline

        res = RAGPipeline().ask(args.question)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, default=str))
        else:
            print(res.answer, "\n")
            for c in res.citations:
                print(f"  [{c.chunk_id}] {c.source} - {c.locator}")
            print(
                f"\n  prompt={res.prompt_version} model={res.model} "
                f"latency={res.stage_timings['total']:.2f}s refused={res.refused}"
            )
        return 0
    if args.cmd == "eval":
        from .eval.run_eval import run

        return run(
            limit=args.limit,
            use_ragas=not args.no_ragas,
            gate=not args.no_gate,
            out=args.out,
            resume=args.resume,
            report_only=args.report_only,
            offline=args.offline,
        )
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("dronedocs.api:app", host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
