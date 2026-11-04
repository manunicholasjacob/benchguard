"""The process's own cgroup, not the root one.

These exist because of a bug the CI dogfood test found before any user did. The probes read
`/sys/fs/cgroup/cpu.stat` directly, which is the ROOT cgroup. A throttled process is almost
never there: `systemd-run --scope`, a container, and most CI runners all put the job in a
descendant, and the root's counters stay at zero while the job is being throttled hard. The
gate would have reported a clean measurement on exactly the machines it was built for.
"""
from __future__ import annotations

import pytest

from benchguard import FAIL, PASS, evaluate, sample_all
from benchguard.probes import CgroupCpuQuota, CgroupCpuThrottle

from test_benchguard import make_machine, write

NESTED = "/system.slice/benchguard-test.slice/run-abc.scope"


def nest(root, *, rel=NESTED, leaf_stat=None, leaf_max=None, root_stat=None, root_max="max"):
    """A machine whose process lives in a descendant cgroup."""
    make_machine(root, thermal=0)
    write(root, "proc/self/cgroup", "0::%s\n" % rel)
    if root_stat is not None:
        write(root, "sys/fs/cgroup/cpu.stat", root_stat)
    if root_max is not None:
        write(root, "sys/fs/cgroup/cpu.max", root_max + "\n")
    base = "sys/fs/cgroup" + rel
    if leaf_stat is not None:
        write(root, base + "/cpu.stat", leaf_stat)
    if leaf_max is not None:
        write(root, base + "/cpu.max", leaf_max + "\n")
    return root


def stat(throttled_usec, nr_throttled=0):
    return ("usage_usec 1\nnr_periods 10\nnr_throttled %d\nthrottled_usec %d\n"
            % (nr_throttled, throttled_usec))


def test_throttling_is_read_from_the_nested_cgroup_not_the_root(tmp_path):
    """The exact failure the dogfood test would have hit: root quiet, leaf throttled."""
    before = nest(str(tmp_path / "b"), root_stat=stat(0), leaf_stat=stat(0))
    after = nest(str(tmp_path / "a"), root_stat=stat(0), leaf_stat=stat(250_000, 7))

    r = CgroupCpuThrottle(after).sample()
    assert r.available and r.value == 250_000, "read the root instead of the leaf"

    v = evaluate(sample_all(before), sample_all(after), duration_s=1.0)
    bad = [f for f in v.findings if f.check == "cgroup_cpu_throttle"]
    assert bad[0].outcome == FAIL
    assert not v.ok()


def test_root_is_still_used_when_there_is_no_proc_self_cgroup(tmp_path):
    """The common case and the fallback: no /proc/self/cgroup, read the root."""
    root = str(tmp_path / "flat")
    make_machine(root, thermal=0, throttled_usec=99, nr_throttled=1)
    assert CgroupCpuThrottle(root).sample().value == 99


def test_a_nested_path_that_does_not_exist_falls_back_to_the_root(tmp_path):
    """A stale or namespaced /proc/self/cgroup must not blind the probe."""
    root = str(tmp_path / "stale")
    make_machine(root, thermal=0, throttled_usec=42, nr_throttled=1)
    write(root, "proc/self/cgroup", "0::/a/path/that/is/not/mounted\n")
    assert CgroupCpuThrottle(root).sample().value == 42


def test_the_tightest_quota_in_the_chain_wins(tmp_path):
    """A leaf saying `max` does not mean unlimited: an ancestor can still bind."""
    root = nest(str(tmp_path / "q"), leaf_max="max", root_max="max")
    write(root, "sys/fs/cgroup/system.slice/cpu.max", "50000 100000\n")   # 0.5 core
    r = CgroupCpuQuota(root).sample()
    assert r.value == pytest.approx(0.5), "walked past the binding ancestor"
    assert r.extra["unlimited"] == 0.0


def test_unlimited_all_the_way_up_is_unlimited(tmp_path):
    root = nest(str(tmp_path / "u"), leaf_max="max", root_max="max")
    write(root, "sys/fs/cgroup/system.slice/cpu.max", "max\n")
    r = CgroupCpuQuota(root).sample()
    assert r.extra["unlimited"] == 1.0 and r.value is None


def test_the_leaf_wins_when_it_is_the_tighter_one(tmp_path):
    root = nest(str(tmp_path / "leaf"), leaf_max="25000 100000", root_max="max")
    write(root, "sys/fs/cgroup/system.slice/cpu.max", "50000 100000\n")
    assert CgroupCpuQuota(root).sample().value == pytest.approx(0.25)


def test_quota_is_unavailable_when_no_level_has_a_cpu_max(tmp_path):
    root = str(tmp_path / "none")
    make_machine(root, thermal=0)
    import os
    os.remove(os.path.join(root, "sys", "fs", "cgroup", "cpu.max"))
    r = CgroupCpuQuota(root).sample()
    assert r.available is False and "ancestors" in r.why_unavailable


def test_threads_over_a_quota_inherited_from_an_ancestor_still_fails(tmp_path):
    """End to end: the quota that binds is two levels up, and the verdict still sees it."""
    root = nest(str(tmp_path / "e"), leaf_max="max", root_max="max")
    write(root, "sys/fs/cgroup/system.slice/cpu.max", "100000 100000\n")  # 1 core
    r = sample_all(root)
    v = evaluate(r, r, duration_s=1.0, threads=4)
    f = next(x for x in v.findings if x.check == "cgroup_cpu_quota")
    assert f.outcome == FAIL and "4 threads" in f.detail


def test_a_quiet_nested_cgroup_still_passes(tmp_path):
    before = nest(str(tmp_path / "qb"), leaf_stat=stat(0), leaf_max="max",
                  root_stat=stat(0))
    after = nest(str(tmp_path / "qa"), leaf_stat=stat(0), leaf_max="max",
                 root_stat=stat(0))
    v = evaluate(sample_all(before), sample_all(after), duration_s=1.0)
    assert next(f for f in v.findings if f.check == "cgroup_cpu_throttle").outcome == PASS
