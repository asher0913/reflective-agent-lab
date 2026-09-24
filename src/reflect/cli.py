"""``reflect-lab compare | sweep | fidelity | report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import compare, error_rate_sweep, model_fidelity


def _compare(args) -> int:
    report = compare(seeds=args.seeds, tasks=args.tasks, error_rate=args.error_rate)
    print("| Strategy | Success | After drift | Tool calls / success | Planner calls / task | Harmful / 100 tasks |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, r in report["strategies"].items():
        cells = [
            name,
            f"{100 * r['success_rate']:.1f}%",
            f"{100 * r['success_first_50_after_drift']:.1f}%",
            f"{r['real_calls_per_success']:.2f}",
            f"{r['planner_calls_per_task']:.2f}",
            f"{r['harmful_per_100_tasks']:.1f}",
        ]
        print("| " + " | ".join(cells) + " |")
    return 0


def _report(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategies.json").write_text(json.dumps(compare(), indent=2) + "\n")
    (out / "error_rate_sweep.json").write_text(json.dumps(error_rate_sweep(), indent=2) + "\n")
    (out / "model_fidelity.json").write_text(json.dumps(model_fidelity(), indent=2) + "\n")
    print(f"wrote {out}/strategies.json, error_rate_sweep.json, model_fidelity.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reflect-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_ = sub.add_parser("compare", help="all strategies on a 500-task stream with tool drift at 250")
    cmp_.add_argument("--seeds", type=int, default=5)
    cmp_.add_argument("--tasks", type=int, default=500)
    cmp_.add_argument("--error-rate", type=float, default=0.35)
    cmp_.set_defaults(func=_compare)
    rep = sub.add_parser("report", help="write every experiment to JSON")
    rep.add_argument("--out", default="results")
    rep.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
