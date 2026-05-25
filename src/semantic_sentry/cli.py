"""Command-line interface for SemanticSentry.

Operates on snapshots already serialised to disk via ``Snapshot.save`` — no
model loading required, so it runs anywhere (CI included). Subcommands:

    semantic-sentry info     SNAPSHOT_DIR
    semantic-sentry compare  BASELINE_DIR CANDIDATE_DIR [--json]
    semantic-sentry gate     BASELINE_DIR CANDIDATE_DIR
                             [--fail-under cka=0.9,nps=0.85]
                             [--fail-over isotropy_delta=0.05] [--json]

``gate`` returns exit code 1 when any threshold is violated, so it drops
straight into a CI step or the bundled GitHub Action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semantic_sentry.core.monitor import DriftMonitor
from semantic_sentry.core.snapshot import Snapshot


def _parse_thresholds(spec: str | None) -> dict[str, float]:
    """Parse ``"cka=0.9,nps=0.85"`` into ``{"cka": 0.9, "nps": 0.85}``."""
    if not spec:
        return {}
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"threshold {part!r} must be METRIC=VALUE"
            )
        name, _, value = part.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"threshold value for {name!r} is not a number: {value!r}"
            ) from exc
    return out


def _load(path: str) -> Snapshot:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: snapshot directory not found: {path}")
    return Snapshot.load(p)


def _cmd_info(args: argparse.Namespace) -> int:
    snap = _load(args.snapshot)
    info = {
        "model_id": snap.model_id,
        "checkpoint_hash": snap.checkpoint_hash,
        "timestamp": snap.timestamp,
        "anchor_set_version": snap.anchor_set_version,
        "tower_count": snap.tower_count,
        "tower_names": list(snap.tower_names),
        "n_samples": int(next(iter(snap.embeddings.values())).shape[0]) if snap.embeddings else 0,
        "metadata": snap.metadata,
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


def _run_compare(args: argparse.Namespace):
    base = _load(args.baseline)
    cand = _load(args.candidate)
    monitor = DriftMonitor()
    return monitor.compare(base, cand)


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = _run_compare(args)
    payload = {
        "severity": comparison.severity.value,
        "global_metrics": dict(comparison.global_metrics),
    }
    if comparison.per_tower_metrics:
        payload["per_tower_metrics"] = comparison.per_tower_metrics
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"severity: {comparison.severity.value}")
        for name, value in comparison.global_metrics.items():
            print(f"  {name}: {value:.4f}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    fail_under = _parse_thresholds(args.fail_under)
    fail_over = _parse_thresholds(args.fail_over)
    if not fail_under and not fail_over:
        raise SystemExit("error: gate needs at least one --fail-under / --fail-over")

    comparison = _run_compare(args)
    metrics = comparison.global_metrics
    violations: list[str] = []

    for name, floor in fail_under.items():
        if name not in metrics:
            violations.append(f"{name}: metric missing (required >= {floor})")
        elif metrics[name] < floor:
            violations.append(f"{name}={metrics[name]:.4f} < {floor} (fail-under)")
    for name, ceil in fail_over.items():
        if name not in metrics:
            violations.append(f"{name}: metric missing (required <= {ceil})")
        elif metrics[name] > ceil:
            violations.append(f"{name}={metrics[name]:.4f} > {ceil} (fail-over)")

    passed = not violations
    if args.json:
        print(json.dumps({
            "passed": passed,
            "severity": comparison.severity.value,
            "global_metrics": dict(metrics),
            "violations": violations,
        }, indent=2))
    else:
        print(f"severity: {comparison.severity.value}")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        if passed:
            print("GATE PASSED")
        else:
            print("GATE FAILED:")
            for v in violations:
                print(f"  - {v}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-sentry",
        description="Semantic drift detection over saved embedding snapshots.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Print a snapshot's metadata.")
    p_info.add_argument("snapshot", help="Snapshot directory.")
    p_info.add_argument("--json", action="store_true", help="Emit JSON.")
    p_info.set_defaults(func=_cmd_info)

    p_cmp = sub.add_parser("compare", help="Compare two snapshots.")
    p_cmp.add_argument("baseline", help="Baseline snapshot directory.")
    p_cmp.add_argument("candidate", help="Candidate snapshot directory.")
    p_cmp.add_argument("--json", action="store_true", help="Emit JSON.")
    p_cmp.set_defaults(func=_cmd_compare)

    p_gate = sub.add_parser(
        "gate",
        help="Compare two snapshots and exit nonzero if thresholds are violated.",
    )
    p_gate.add_argument("baseline", help="Baseline snapshot directory.")
    p_gate.add_argument("candidate", help="Candidate snapshot directory.")
    p_gate.add_argument(
        "--fail-under",
        metavar="METRIC=MIN[,...]",
        help="Fail if a metric drops below MIN (e.g. cka=0.9,nps=0.85).",
    )
    p_gate.add_argument(
        "--fail-over",
        metavar="METRIC=MAX[,...]",
        help="Fail if a metric exceeds MAX (e.g. isotropy_delta=0.05).",
    )
    p_gate.add_argument("--json", action="store_true", help="Emit JSON.")
    p_gate.set_defaults(func=_cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
