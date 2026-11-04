"""Command line: wrap a benchmark command and judge the machine it ran on.

    benchguard run --threads 4 -- ./my-benchmark --fast
    benchguard sample            # what can this machine even tell us?

`run` returns the command's own exit code when the measurement was valid, and a distinct code
when it was not, so a CI job can tell "the benchmark failed" from "the benchmark's number is
not trustworthy". Those are different problems and conflating them is how a flaky runner gets
mistaken for a regression.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

from .probes import sample_all
from .verdict import evaluate

EXIT_INVALID = 90          # the measurement happened but cannot be trusted
EXIT_NOT_PROVEN = 91       # strict mode, and the checks could not be made here


def _emit(verdict, args) -> None:
    if args.json:
        text = verdict.to_json()
        if args.json == "-":
            print(text)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
            print("benchguard: wrote %s" % args.json, file=sys.stderr)
    if not args.quiet:
        for f in verdict.findings:
            mark = {"pass": "ok  ", "fail": "FAIL", "unknown": "? "}.get(f.outcome, "?")
            print("benchguard: %-4s %-22s %s" % (mark, f.check, f.detail), file=sys.stderr)
        print("benchguard: %s" % verdict.summary(strict=args.strict), file=sys.stderr)

    # GitHub Actions surfaces these as annotations on the job
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for f in verdict.failures:
            print("::error title=benchguard %s::%s" % (f.check, f.detail))
        if args.strict:
            for f in verdict.unknowns:
                print("::warning title=benchguard %s::%s" % (f.check, f.detail))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            try:
                with open(summary, "a", encoding="utf-8") as fh:
                    fh.write("### benchguard\n\n%s\n\n" % verdict.summary(strict=args.strict))
                    fh.write("| check | outcome | detail |\n|---|---|---|\n")
                    for f in verdict.findings:
                        fh.write("| %s | %s | %s |\n" % (f.check, f.outcome, f.detail))
                    fh.write("\n")
            except OSError:
                pass


def cmd_run(args) -> int:
    if not args.command:
        print("benchguard: nothing to run; put the command after --", file=sys.stderr)
        return 2
    before = sample_all(args.root)
    t0 = time.perf_counter()
    completed = subprocess.run(args.command)
    duration = time.perf_counter() - t0
    after = sample_all(args.root)
    verdict = evaluate(before, after, duration_s=duration, threads=args.threads,
                       competing_fraction=args.competing_fraction)
    _emit(verdict, args)

    if completed.returncode != 0:
        return completed.returncode          # the command's own failure comes first
    if verdict.failures:
        return EXIT_INVALID
    if args.strict and verdict.unknowns:
        return EXIT_NOT_PROVEN
    return 0


def cmd_sample(args) -> int:
    readings = sample_all(args.root)
    for name, r in readings.items():
        if r.available:
            val = "unlimited" if r.value is None else ("%g %s" % (r.value, r.unit))
            print("%-22s %s%s" % (name, val, "  [%s]" % r.raw if r.raw else ""))
        else:
            print("%-22s unavailable: %s" % (name, r.why_unavailable))
    missing = [r for r in readings.values() if not r.available]
    if missing:
        print("\n%d of %d checks cannot be made here. In strict mode that is not a pass."
              % (len(missing), len(readings)))
    return 0


def _common() -> argparse.ArgumentParser:
    """Options that make sense for every subcommand.

    Added to the top-level parser AND to each subparser as a parent, because argparse
    otherwise accepts them only before the subcommand, and `benchguard run --root /x` is the
    order anyone would actually type.
    """
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--root", default="/", help="filesystem root to read state from")
    c.add_argument("--json", metavar="PATH",
                   help="write the provenance record here, or - for stdout")
    c.add_argument("--strict", action="store_true",
                   help="treat checks that could not be made as failures")
    c.add_argument("--quiet", action="store_true", help="suppress the per-check lines")
    c.add_argument("--competing-fraction", type=float, default=0.10, dest="competing_fraction",
                   help="share of machine busy time allowed to other processes (default 0.10)")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    p = argparse.ArgumentParser(prog="benchguard", parents=[common],
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common],
                       help="run a command and judge the machine around it")
    r.add_argument("--threads", type=int, default=None,
                   help="threads the benchmark will use, to compare against the CPU quota")
    r.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sample", parents=[common],
                       help="show what this machine can report")
    s.set_defaults(func=cmd_sample)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
