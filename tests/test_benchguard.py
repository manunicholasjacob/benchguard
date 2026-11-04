"""Tests for benchguard.

The fixture trees are built in-test rather than committed, so that each test states the exact
before and after state it is asserting about. That matters more here than usual: this package's
whole job is to notice a difference between two samples, so a test that cannot show you both
samples is not testing the thing.

These run on any platform, including Windows, because every probe reads from a relocatable root.
"""
from __future__ import annotations

import os

import pytest

from benchguard import FAIL, PASS, UNKNOWN, evaluate, sample_all
from benchguard.probes import (CgroupCpuQuota, CgroupCpuThrottle, CompetingLoad,
                               MajorFaults, ScalingGovernor, ThermalThrottle)


def write(root, rel, text):
    path = os.path.join(root, rel.lstrip("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def make_machine(root, *, throttled_usec=0, nr_throttled=0, quota="max 100000",
                 thermal=None, governor="performance", majflt=0, child_majflt=0,
                 cpu_busy=1000.0, cpu_idle=9000.0, own_busy=1000.0, child_busy=0.0,
                 ncpu=2):
    """A whole plausible machine under `root`. Every argument is something a test varies."""
    write(root, "sys/fs/cgroup/cpu.stat",
          "usage_usec 123456\nuser_usec 100000\nsystem_usec 23456\n"
          "nr_periods 100\nnr_throttled %d\nthrottled_usec %d\n" % (nr_throttled, throttled_usec))
    write(root, "sys/fs/cgroup/cpu.max", quota + "\n")
    if thermal is not None:
        for cpu in range(ncpu):
            write(root, "sys/devices/system/cpu/cpu%d/thermal_throttle/core_throttle_count" % cpu,
                  "%d\n" % thermal)
    for cpu in range(ncpu):
        gov = governor[cpu] if isinstance(governor, (list, tuple)) else governor
        write(root, "sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor" % cpu, gov + "\n")
    # /proc/self/stat is "pid (comm) state" then the rest. Splitting past the comm keeps the
    # state letter first, so fields[k] is stat field 3+k: majflt (12) is fields[9], utime and
    # stime (14, 15) are fields[11] and [12]. `tail` is written after the state letter, so its
    # index is one lower again. Verified against proc(5).
    tail = [0] * 30
    tail[8] = majflt              # -> fields[9]  -> stat field 12 (majflt)
    tail[9] = int(child_majflt)   # -> fields[10] -> stat field 13 (cmajflt)
    tail[10] = int(own_busy)      # -> fields[11] -> stat field 14 (utime)
    tail[11] = 0                  # -> fields[12] -> stat field 15 (stime)
    tail[12] = int(child_busy)    # -> fields[13] -> stat field 16 (cutime)
    tail[13] = 0                  # -> fields[14] -> stat field 17 (cstime)
    write(root, "proc/self/stat",
          "1234 (python) R " + " ".join(str(x) for x in tail) + "\n")
    # user nice system idle iowait irq softirq
    write(root, "proc/stat",
          "cpu  %d 0 0 %d 0 0 0\ncpu0 1 0 0 1 0 0 0\n" % (int(cpu_busy), int(cpu_idle)))


@pytest.fixture
def clean(tmp_path):
    root = str(tmp_path / "clean")
    make_machine(root, thermal=0)
    return root


# ---------------------------------------------------------------- probes ----------------

def test_probes_read_a_clean_machine(clean):
    r = sample_all(clean)
    assert r["cgroup_cpu_throttle"].available and r["cgroup_cpu_throttle"].value == 0
    assert r["cgroup_cpu_quota"].extra["unlimited"] == 1.0
    assert r["thermal_throttle"].value == 0
    assert r["scaling_governor"].raw == "performance"
    assert r["major_faults"].value == 0
    assert r["competing_load"].value == 1000.0


def test_absent_files_are_unavailable_not_zero(tmp_path):
    """The distinction this package turns on: not looked at is not the same as fine."""
    r = sample_all(str(tmp_path / "empty"))
    for name in ("cgroup_cpu_throttle", "cgroup_cpu_quota", "thermal_throttle",
                 "scaling_governor", "major_faults", "competing_load"):
        assert r[name].available is False, name
        assert r[name].why_unavailable, "%s must say why" % name
        assert r[name].value is None


def test_quota_is_reported_in_cores(tmp_path):
    root = str(tmp_path / "q")
    make_machine(root, quota="50000 100000")
    assert CgroupCpuQuota(root).sample().value == pytest.approx(0.5)


def test_unparseable_quota_is_unavailable(tmp_path):
    root = str(tmp_path / "bad")
    make_machine(root, quota="garbage")
    assert CgroupCpuQuota(root).sample().available is False


def test_majflt_survives_a_comm_containing_spaces_and_parens(tmp_path):
    """A process named "(my benchmark)" breaks naive splitting of /proc/self/stat."""
    root = str(tmp_path / "comm")
    make_machine(root)
    tail = [0] * 30
    tail[8] = 77
    write(root, "proc/self/stat",
          "1234 ((my benchmark) R) R " + " ".join(str(x) for x in tail) + "\n")
    assert MajorFaults(root).sample().value == 77


# ---------------------------------------------------------------- verdicts --------------

def outcome(v, check):
    return next(f.outcome for f in v.findings if f.check == check)


def two_samples(tmp_path, name, before_kw, after_kw):
    b_root = str(tmp_path / (name + "_b"))
    a_root = str(tmp_path / (name + "_a"))
    make_machine(b_root, **before_kw)
    make_machine(a_root, **after_kw)
    return sample_all(b_root), sample_all(a_root)


def test_clean_machine_passes(clean, tmp_path):
    # Two distinct samples on purpose. Comparing one sample to itself leaves zero busy jiffies,
    # which the competing-load check correctly refuses to judge, so the strict verdict would be
    # UNKNOWN for a reason that has nothing to do with the machine being clean.
    before, after = two_samples(tmp_path, "cleanpair",
                                dict(thermal=0, cpu_busy=1000.0, own_busy=1000.0),
                                dict(thermal=0, cpu_busy=2000.0, own_busy=2000.0))
    v = evaluate(before, after, duration_s=1.0, threads=1)
    assert v.ok() and v.ok(strict=True), v.summary()
    assert not v.failures


def test_cpu_throttling_fails_and_quantifies_itself(tmp_path):
    before, after = two_samples(
        tmp_path, "thr",
        dict(thermal=0, throttled_usec=0, nr_throttled=0),
        dict(thermal=0, throttled_usec=250000, nr_throttled=5))
    v = evaluate(before, after, duration_s=1.0)
    assert outcome(v, "cgroup_cpu_throttle") == FAIL
    detail = v.failures[0].detail
    assert "250000 us" in detail.replace(",", "") or "250000" in detail
    assert "25.0%" in detail, detail      # 0.25 s of a 1 s measurement
    assert not v.ok()


def test_threads_over_quota_fails_even_without_throttling_yet(tmp_path):
    root = str(tmp_path / "overcommit")
    make_machine(root, thermal=0, quota="100000 100000")   # 1 core
    r = sample_all(root)
    v = evaluate(r, r, duration_s=1.0, threads=4)
    assert outcome(v, "cgroup_cpu_quota") == FAIL
    assert "4 threads" in v.failures[0].detail


def test_thermal_throttling_fails(tmp_path):
    before, after = two_samples(tmp_path, "therm", dict(thermal=0), dict(thermal=3))
    v = evaluate(before, after, duration_s=1.0)
    assert outcome(v, "thermal_throttle") == FAIL


def test_major_faults_fail(tmp_path):
    before, after = two_samples(tmp_path, "mf", dict(thermal=0, majflt=0),
                                dict(thermal=0, majflt=11404))
    v = evaluate(before, after, duration_s=1.0)
    assert outcome(v, "major_faults") == FAIL
    assert "11404" in v.failures[0].detail


def test_competing_load_fails_above_the_threshold(tmp_path):
    """32% of busy time belonging to someone else is the case this check was built for."""
    before, after = two_samples(
        tmp_path, "busy",
        dict(thermal=0, cpu_busy=1000.0, own_busy=1000.0),
        dict(thermal=0, cpu_busy=2000.0, own_busy=1680.0))   # 320 of 1000 = 32%
    v = evaluate(before, after, duration_s=1.0)
    assert outcome(v, "competing_load") == FAIL
    assert "32%" in v.failures[0].detail


def test_competing_load_tolerates_ordinary_daemon_noise(tmp_path):
    before, after = two_samples(
        tmp_path, "quiet",
        dict(thermal=0, cpu_busy=1000.0, own_busy=1000.0),
        dict(thermal=0, cpu_busy=2000.0, own_busy=1960.0))   # 4%
    v = evaluate(before, after, duration_s=1.0)
    assert outcome(v, "competing_load") == PASS


def test_mixed_governors_fail(tmp_path):
    root = str(tmp_path / "mixedgov")
    make_machine(root, thermal=0, governor=["performance", "powersave"])
    r = sample_all(root)
    v = evaluate(r, r, duration_s=1.0)
    assert outcome(v, "scaling_governor") == FAIL


def test_powersave_is_unknown_not_failure(tmp_path):
    """It does not invalidate the number; it means the clock is not yours to explain."""
    root = str(tmp_path / "ps")
    make_machine(root, thermal=0, governor="powersave")
    r = sample_all(root)
    v = evaluate(r, r, duration_s=1.0)
    assert outcome(v, "scaling_governor") == UNKNOWN
    assert v.ok() and not v.ok(strict=True)


def test_unknown_is_not_a_pass(tmp_path):
    """A machine where nothing can be checked must not report a clean bill of health."""
    r = sample_all(str(tmp_path / "nothing"))
    v = evaluate(r, r, duration_s=1.0)
    assert v.ok() is True            # nothing failed
    assert v.ok(strict=True) is False  # but nothing was proven either
    assert len(v.unknowns) == len(v.findings)
    assert "NOT PROVEN" in v.summary(strict=True)


def test_short_measurement_cannot_judge_competing_load(tmp_path):
    root = str(tmp_path / "short")
    make_machine(root, thermal=0, cpu_busy=0.0, cpu_idle=0.0, own_busy=0.0)
    r = sample_all(root)
    v = evaluate(r, r, duration_s=0.0001)
    assert outcome(v, "competing_load") == UNKNOWN


# ---------------------------------------------------------------- reporting -------------

def test_every_finding_explains_itself(tmp_path):
    before, after = two_samples(tmp_path, "expl", dict(thermal=0),
                                dict(thermal=2, throttled_usec=1000, nr_throttled=1, majflt=5))
    v = evaluate(before, after, duration_s=1.0)
    for f in v.findings:
        assert f.detail.strip(), f.check
        assert f.outcome in (PASS, FAIL, UNKNOWN)


def test_json_round_trips_and_carries_provenance(clean):
    import json
    r = sample_all(clean)
    v = evaluate(r, r, duration_s=0.5, threads=2)
    d = json.loads(v.to_json())
    assert d["provenance"]["threads_declared"] == 2
    assert d["provenance"]["platform"]
    assert len(d["findings"]) == len(v.findings)


def test_guard_context_manager_measures_around_the_work(clean):
    from benchguard import guard
    with guard(root=clean, threads=1) as g:
        sum(range(10000))
    assert g.verdict is not None
    assert g.verdict.duration_s > 0
    assert g.verdict.ok()
