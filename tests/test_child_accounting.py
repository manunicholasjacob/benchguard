"""The work is usually in a child process, and the probes have to see it.

`benchguard run` forks the benchmark and spends almost no CPU itself. Reading utime and stime
alone, benchguard therefore concluded that ~100% of the machine's busy time belonged to
somebody else and rejected the measurement as invalid. Every CLI run on a real machine was
affected: the one use the package is named for.

Nothing in the unit suite caught it, because the fake machines wrote the numbers the probe
expected to read. The first CI run of the Action on a quiet hosted runner caught it in about
forty seconds. These tests are the version that would have caught it earlier: they describe a
machine where the work was done in a reaped child, which is what the CLI actually produces.

The real end-to-end check is at the bottom, and it only runs on Linux, because the counters
do not exist anywhere else.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from benchguard.probes import sample_all
from benchguard.verdict import FAIL, PASS, evaluate

from test_benchguard import make_machine


def outcome(verdict, check):
    return next(f.outcome for f in verdict.findings if f.check == check)


def around(root, before, after):
    make_machine(root, **before)
    b = sample_all(root)
    make_machine(root, **after)
    return evaluate(b, sample_all(root), duration_s=1.0)


BASE = dict(thermal=0, cpu_busy=1000.0, own_busy=100.0, child_busy=0.0)


def test_a_measurement_run_entirely_in_a_child_is_credited_to_us(tmp_path):
    """100 busy jiffies on the machine, all 100 burned by the benchmark child.

    This is the shape of every `benchguard run`. It must pass.
    """
    v = around(str(tmp_path / "m"),
               dict(BASE),
               dict(BASE, cpu_busy=1100.0, child_busy=100.0))
    assert outcome(v, "competing_load") == PASS, [f.detail for f in v.findings]
    assert v.ok()


def test_the_same_machine_without_the_child_credit_still_fails(tmp_path):
    """The discriminating half: if those 100 jiffies really were somebody else's, say so.

    Without this, the test above would pass for a probe that credited everything to us, and
    the check would be decorative.
    """
    v = around(str(tmp_path / "m"),
               dict(BASE),
               dict(BASE, cpu_busy=1100.0, child_busy=0.0))
    assert outcome(v, "competing_load") == FAIL
    assert not v.ok()


def test_a_child_that_burned_only_part_of_the_busy_time_is_judged_on_the_rest(tmp_path):
    """80 of 100 jiffies ours, 20 somebody else's: 20% is over the 10% default."""
    v = around(str(tmp_path / "m"),
               dict(BASE),
               dict(BASE, cpu_busy=1100.0, child_busy=80.0))
    assert outcome(v, "competing_load") == FAIL
    # and inside the allowance it passes
    v2 = around(str(tmp_path / "m2"),
                dict(BASE),
                dict(BASE, cpu_busy=1100.0, child_busy=95.0))
    assert outcome(v2, "competing_load") == PASS


def test_the_record_separates_our_own_cpu_from_our_childrens(tmp_path):
    """Both numbers are kept, because "the benchmark used no CPU itself" is worth seeing."""
    root = str(tmp_path / "m")
    make_machine(root, **dict(BASE, own_busy=7.0, child_busy=93.0))
    r = sample_all(root)["competing_load"]
    assert r.extra["self_busy"] == 7.0
    assert r.extra["child_busy"] == 93.0
    assert r.extra["own_busy"] == 100.0


def test_major_faults_in_a_child_are_not_invisible(tmp_path):
    """cmajflt, for the same reason: the CLI's faults are all in the child."""
    v = around(str(tmp_path / "m"),
               dict(BASE, majflt=0, child_majflt=0),
               dict(BASE, majflt=0, child_majflt=4096,
                    cpu_busy=1100.0, child_busy=100.0))
    assert outcome(v, "major_faults") == FAIL
    detail = next(f.detail for f in v.findings if f.check == "major_faults")
    assert "4096" in detail


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/stat is Linux-only")
def test_the_cli_does_not_reject_a_real_quiet_measurement():
    """End to end on this machine, no fixtures: the check that CI had to find for us.

    A short CPU burn in a child on an otherwise quiet machine must not be reported as
    contaminated. If this fails on a busy machine that is a true positive, so the assertion
    names the competing-load finding specifically rather than requiring a clean verdict.
    """
    r = subprocess.run(
        [sys.executable, "-m", "benchguard.cli", "run", "--threads", "1", "--",
         sys.executable, "-c", "sum(i*i for i in range(20_000_000))"],
        capture_output=True, text=True)
    assert "100% of machine busy time" not in r.stderr, r.stderr
    assert r.returncode in (0, 90), r.stderr
