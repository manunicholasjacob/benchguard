"""Turn two samples into a verdict, with the reason attached.

The design rule here is that a gate which cannot explain itself is not usable in CI. Every
outcome carries the numbers it was computed from, so a failing job tells you which counter
moved and by how much, not that "the environment was unstable".

Three outcomes, and the third is the one most tools get wrong:

    PASS        nothing moved that should not have
    FAIL        something moved that invalidates the number
    UNKNOWN     the check could not be made here

UNKNOWN is deliberately not PASS. A CI job on a platform without cgroup v2 has not proved its
measurement was unthrottled; it has failed to look. Whether that should stop the build is the
caller's policy, which is why `Verdict.ok` takes `strict`.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .probes import Reading, sample_all

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


@dataclass
class Finding:
    check: str
    outcome: str
    detail: str
    delta: Optional[float] = None
    unit: str = ""


@dataclass
class Verdict:
    findings: List[Finding] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def failures(self) -> List[Finding]:
        return [f for f in self.findings if f.outcome == FAIL]

    @property
    def unknowns(self) -> List[Finding]:
        return [f for f in self.findings if f.outcome == UNKNOWN]

    def ok(self, strict: bool = False) -> bool:
        """True when the measurement is usable. `strict` also rejects checks that could not run."""
        if self.failures:
            return False
        return not (strict and self.unknowns)

    def summary(self, strict: bool = False) -> str:
        if self.failures:
            return "INVALID: " + "; ".join(f.detail for f in self.failures)
        if strict and self.unknowns:
            return "NOT PROVEN: " + "; ".join(f.detail for f in self.unknowns)
        tail = " (%d check%s could not run)" % (
            len(self.unknowns), "" if len(self.unknowns) == 1 else "s") if self.unknowns else ""
        return "valid: %d check%s passed%s" % (
            len([f for f in self.findings if f.outcome == PASS]),
            "" if len([f for f in self.findings if f.outcome == PASS]) == 1 else "s", tail)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"findings": [asdict(f) for f in self.findings],
             "provenance": self.provenance,
             "duration_s": round(self.duration_s, 6)},
            indent=indent, sort_keys=True)


def _delta(before: Reading, after: Reading) -> Optional[float]:
    if before.available and after.available and before.value is not None and after.value is not None:
        return after.value - before.value
    return None


def evaluate(before: Dict[str, Reading], after: Dict[str, Reading],
             duration_s: float = 0.0, threads: Optional[int] = None,
             competing_fraction: float = 0.10) -> Verdict:
    """Compare two samples taken around a measurement.

    `competing_fraction` is the share of machine busy time that may belong to other processes
    before the measurement is called contaminated. It defaults to 10%, well under the 32% that
    a single background compilation job cost in the case this check exists for, and above the
    couple of percent that normal daemon noise produces.
    """
    v = Verdict(duration_s=duration_s)

    def add(check, outcome, detail, delta=None, unit=""):
        v.findings.append(Finding(check=check, outcome=outcome, detail=detail,
                                  delta=delta, unit=unit))

    # -- CPU quota throttling ------------------------------------------------------------
    b, a = before["cgroup_cpu_throttle"], after["cgroup_cpu_throttle"]
    if not a.available:
        add("cgroup_cpu_throttle", UNKNOWN, "CPU throttling not checked: " + a.why_unavailable)
    else:
        d = _delta(b, a) or 0.0
        n = (a.extra.get("nr_throttled", 0.0) - b.extra.get("nr_throttled", 0.0))
        if d > 0:
            share = (d / 1e6) / duration_s * 100 if duration_s > 0 else float("nan")
            add("cgroup_cpu_throttle", FAIL,
                "the cgroup was CPU-throttled for %.0f us across %.0f periods during the "
                "measurement (%.1f%% of its wall clock)" % (d, n, share), d, "us")
        else:
            add("cgroup_cpu_throttle", PASS, "no CPU quota throttling during the measurement", 0.0, "us")

    # -- quota versus threads ------------------------------------------------------------
    q = after["cgroup_cpu_quota"]
    if not q.available:
        add("cgroup_cpu_quota", UNKNOWN, "CPU quota not read: " + q.why_unavailable)
    elif q.extra.get("unlimited", 0.0) == 1.0:
        add("cgroup_cpu_quota", PASS, "no CPU quota set")
    elif threads is not None and q.value is not None and threads > q.value:
        add("cgroup_cpu_quota", FAIL,
            "%d threads under a %.2f core quota: the thread count exceeds the quota, which is "
            "the configuration that produces throttling" % (threads, q.value), q.value, "cores")
    else:
        add("cgroup_cpu_quota", PASS, "CPU quota %.2f cores" % (q.value or 0.0), q.value, "cores")

    # -- thermal -------------------------------------------------------------------------
    b, a = before["thermal_throttle"], after["thermal_throttle"]
    if not a.available:
        add("thermal_throttle", UNKNOWN, "thermal throttling not checked: " + a.why_unavailable)
    else:
        d = _delta(b, a) or 0.0
        if d > 0:
            add("thermal_throttle", FAIL,
                "thermal throttle counters rose by %.0f during the measurement" % d, d, "events")
        else:
            add("thermal_throttle", PASS, "no thermal throttling during the measurement", 0.0, "events")

    # -- governor ------------------------------------------------------------------------
    g = after["scaling_governor"]
    if not g.available:
        add("scaling_governor", UNKNOWN, "governor not read: " + g.why_unavailable)
    elif g.value is not None and g.value > 1:
        add("scaling_governor", FAIL,
            "cores disagree on the scaling governor (%s), so the clock is not a property of the "
            "workload" % g.raw)
    elif g.raw and g.raw not in ("performance", "userspace"):
        add("scaling_governor", UNKNOWN,
            "governor is %r, so the achieved clock is scheduler-dependent; pin it to performance "
            "for a comparable number" % g.raw)
    else:
        add("scaling_governor", PASS, "governor is %r" % g.raw)

    # -- major faults --------------------------------------------------------------------
    b, a = before["major_faults"], after["major_faults"]
    if not a.available:
        add("major_faults", UNKNOWN, "major faults not checked: " + a.why_unavailable)
    else:
        d = _delta(b, a) or 0.0
        if d > 0:
            add("major_faults", FAIL,
                "%.0f major page faults during the measurement: it waited on storage, so it is "
                "timing the page cache as well as the work" % d, d, "faults")
        else:
            add("major_faults", PASS, "no major page faults during the measurement", 0.0, "faults")

    # -- competing load ------------------------------------------------------------------
    b, a = before["competing_load"], after["competing_load"]
    if not a.available:
        add("competing_load", UNKNOWN, "competing load not checked: " + a.why_unavailable)
    else:
        busy = _delta(b, a) or 0.0
        own = a.extra.get("own_busy", 0.0) - b.extra.get("own_busy", 0.0)
        other = max(0.0, busy - own)
        frac = (other / busy) if busy > 0 else 0.0
        if busy <= 0:
            add("competing_load", UNKNOWN, "no busy jiffies accumulated; measurement too short to judge")
        elif frac > competing_fraction:
            add("competing_load", FAIL,
                "%.0f%% of machine busy time during the measurement belonged to other processes "
                "(%.0f of %.0f jiffies)" % (frac * 100, other, busy), frac, "fraction")
        else:
            add("competing_load", PASS,
                "%.0f%% of busy time was other processes" % (frac * 100), frac, "fraction")

    v.provenance = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "threads_declared": threads,
        "readings_after": {k: {"value": r.value, "unit": r.unit, "available": r.available,
                               "why_unavailable": r.why_unavailable}
                           for k, r in after.items()},
    }
    return v


class guard:
    """Context manager: sample, run the measurement, sample again, judge.

    Sampling happens around the work rather than before it, because the state that matters is
    the state during the measurement. A machine checked beforehand and found quiet can still
    start compiling something the moment the benchmark begins, which is exactly what happened
    in the case the competing-load check exists for.

        with guard(threads=4) as g:
            result = benchmark()
        if not g.verdict.ok():
            raise SystemExit(g.verdict.summary())
    """

    def __init__(self, root: str = "/", threads: Optional[int] = None,
                 competing_fraction: float = 0.10):
        self.root, self.threads = root, threads
        self.competing_fraction = competing_fraction
        self.before: Dict[str, Reading] = {}
        self.after: Dict[str, Reading] = {}
        self.verdict: Optional[Verdict] = None
        self._t0 = 0.0

    def __enter__(self) -> "guard":
        self.before = sample_all(self.root)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = time.perf_counter() - self._t0
        self.after = sample_all(self.root)
        self.verdict = evaluate(self.before, self.after, duration_s=duration,
                                threads=self.threads,
                                competing_fraction=self.competing_fraction)
        return False
