"""Readers for the machine state that decides whether a measurement meant anything.

Every probe reads from a filesystem root rather than hard-coding "/". That is not an
abstraction for its own sake: it is the only way to test this package on a machine that is not
the one it describes, and the fixture trees under tests/fixtures are real captures.

Probes never raise for an absent file. A check that cannot be made is reported as unavailable
and says why, because a gate that silently skips is worse than no gate: it reports a pass it
did not earn.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Reading:
    """One sample of one thing, or the reason there isn't one."""

    name: str
    value: Optional[float] = None
    unit: str = ""
    raw: Optional[str] = None
    available: bool = True
    why_unavailable: str = ""
    extra: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def missing(cls, name: str, why: str) -> "Reading":
        return cls(name=name, available=False, why_unavailable=why)


class Probe:
    """Base class. `root` lets the whole tree be relocated for tests and for containers."""

    name = "probe"

    def __init__(self, root: str = "/"):
        self.root = root

    # -- small filesystem helpers, all of them forgiving --------------------------------
    def _path(self, *parts: str) -> str:
        return os.path.join(self.root, *[p.lstrip("/") for p in parts])

    def _read(self, *parts: str) -> Optional[str]:
        try:
            with open(self._path(*parts), "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except (OSError, ValueError):
            return None

    def _glob_dirs(self, rel: str) -> List[str]:
        import glob
        return sorted(glob.glob(self._path(rel)))

    def sample(self) -> Reading:  # pragma: no cover - interface
        raise NotImplementedError


class CgroupCpuThrottle(Probe):
    """cgroup v2 `cpu.stat`: how long the kernel refused to schedule us.

    This is the check with the largest measured magnitude behind it. A CPU quota that binds
    costs roughly half of throughput on Cortex-A76 and roughly two thirds on x86, and the
    throttle counters rise monotonically with the loss, so a non-zero delta here is not a
    warning about a possible effect, it is the effect.
    """

    name = "cgroup_cpu_throttle"

    def sample(self) -> Reading:
        text = self._read("sys/fs/cgroup/cpu.stat")
        if text is None:
            return Reading.missing(self.name, "no /sys/fs/cgroup/cpu.stat (not cgroup v2, or not Linux)")
        fields = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    fields[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        if "throttled_usec" not in fields:
            return Reading.missing(self.name, "cpu.stat has no throttled_usec; CONFIG_CFS_BANDWIDTH off")
        return Reading(
            name=self.name,
            value=fields["throttled_usec"],
            unit="us",
            raw=text.strip(),
            extra={k: v for k, v in fields.items() if k in ("nr_throttled", "nr_periods", "usage_usec")},
        )


class CgroupCpuQuota(Probe):
    """cgroup v2 `cpu.max`: the quota itself, in cores.

    Static, so it is read once rather than differenced. Reported because a quota smaller than
    the thread count is the configuration that produces the throttling above, and knowing the
    ratio turns "it was throttled" into "it was throttled because you asked for more threads
    than cores".
    """

    name = "cgroup_cpu_quota"

    def sample(self) -> Reading:
        text = self._read("sys/fs/cgroup/cpu.max")
        if text is None:
            return Reading.missing(self.name, "no /sys/fs/cgroup/cpu.max")
        parts = text.split()
        if not parts:
            return Reading.missing(self.name, "cpu.max empty")
        if parts[0] == "max":
            return Reading(name=self.name, value=None, unit="cores", raw=text.strip(),
                           extra={"unlimited": 1.0})
        try:
            quota, period = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return Reading.missing(self.name, "cpu.max unparseable: %r" % text.strip())
        if period <= 0:
            return Reading.missing(self.name, "cpu.max period is zero")
        return Reading(name=self.name, value=quota / period, unit="cores", raw=text.strip(),
                       extra={"unlimited": 0.0})


class ThermalThrottle(Probe):
    """Per-core and per-package thermal throttle counters.

    x86 exposes these directly. Many Arm boards do not, which is a real limitation rather than
    a gap to paper over: on a Raspberry Pi the equivalent lives behind `vcgencmd get_throttled`
    and on Android the only sensor reachable without root moved in the opposite direction to
    the load for an entire six-minute window, so an absent counter here is reported as absent.
    """

    name = "thermal_throttle"

    def sample(self) -> Reading:
        total = 0.0
        found = 0
        per_cpu: Dict[str, float] = {}
        for d in self._glob_dirs("sys/devices/system/cpu/cpu*/thermal_throttle"):
            for leaf in ("core_throttle_count", "package_throttle_count"):
                try:
                    with open(os.path.join(d, leaf), encoding="utf-8") as fh:
                        v = float(fh.read().strip())
                except (OSError, ValueError):
                    continue
                found += 1
                total += v
                cpu = os.path.basename(os.path.dirname(d))
                per_cpu["%s.%s" % (cpu, leaf)] = v
        if not found:
            return Reading.missing(
                self.name,
                "no thermal_throttle counters (common on Arm; use vcgencmd on a Pi)")
        return Reading(name=self.name, value=total, unit="events", extra=per_cpu)


class ScalingGovernor(Probe):
    """The governor, and whether the cores disagree about it.

    Read once. A `powersave` governor does not invalidate a measurement by itself, but it makes
    the achieved clock a function of the scheduler's mood rather than of the work, which is the
    condition under which a forced-clock sweep showed a 37.5% clock cut costing 29 to 39%.
    """

    name = "scaling_governor"

    def sample(self) -> Reading:
        govs: Dict[str, float] = {}
        names = []
        for d in self._glob_dirs("sys/devices/system/cpu/cpu[0-9]*/cpufreq"):
            try:
                with open(os.path.join(d, "scaling_governor"), encoding="utf-8") as fh:
                    names.append(fh.read().strip())
            except OSError:
                continue
        if not names:
            return Reading.missing(self.name, "no cpufreq/scaling_governor (no cpufreq driver, or a VM)")
        for n in names:
            govs[n] = govs.get(n, 0.0) + 1.0
        return Reading(name=self.name, value=float(len(govs)), unit="distinct",
                       raw=",".join(sorted(govs)), extra=govs)


class MajorFaults(Probe):
    """Major page faults taken by this process: field 12 of /proc/self/stat.

    Major faults mean the measurement waited on storage. On a 272,000-page model a run that
    took 11,404 of them is not measuring the model, it is measuring the page cache, and the
    difference does not show up in wall-clock variance in any way an eye can catch.
    """

    name = "major_faults"

    def sample(self) -> Reading:
        text = self._read("proc/self/stat")
        if text is None:
            return Reading.missing(self.name, "no /proc/self/stat")
        # the comm field can contain spaces and parentheses, so split after the last ')'
        close = text.rfind(")")
        if close < 0:
            return Reading.missing(self.name, "/proc/self/stat has no comm field")
        fields = text[close + 2:].split()
        # after comm and state, majflt is the 10th remaining field (stat field 12)
        try:
            majflt = float(fields[9])
        except (IndexError, ValueError):
            return Reading.missing(self.name, "/proc/self/stat too short for majflt")
        return Reading(name=self.name, value=majflt, unit="faults")


class CompetingLoad(Probe):
    """Busy jiffies on the machine that were not spent on this process.

    This is the check that comes from the most expensive lesson in the set. A phone left idle
    and charging ran background application compilation during the measurement, cost 32% of a
    memory-bandwidth ceiling, and every scatter-based health check on the run passed. Load
    average is useless for this because it is a decayed average; the process table is not.
    """

    name = "competing_load"

    def sample(self) -> Reading:
        stat = self._read("proc/stat")
        own = self._read("proc/self/stat")
        if stat is None or own is None:
            return Reading.missing(self.name, "no /proc/stat or /proc/self/stat")
        cpu_line = next((l for l in stat.splitlines() if l.startswith("cpu ")), None)
        if cpu_line is None:
            return Reading.missing(self.name, "/proc/stat has no aggregate cpu line")
        try:
            vals = [float(v) for v in cpu_line.split()[1:]]
        except ValueError:
            return Reading.missing(self.name, "/proc/stat cpu line unparseable")
        if len(vals) < 4:
            return Reading.missing(self.name, "/proc/stat cpu line too short")
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)   # idle + iowait
        busy = sum(vals) - idle
        close = own.rfind(")")
        fields = own[close + 2:].split() if close >= 0 else []
        try:
            own_busy = float(fields[11]) + float(fields[12])   # utime + stime
        except (IndexError, ValueError):
            return Reading.missing(self.name, "/proc/self/stat too short for utime/stime")
        return Reading(name=self.name, value=busy, unit="jiffies",
                       extra={"own_busy": own_busy, "idle": idle})


ALL_PROBES = [
    CgroupCpuThrottle,
    CgroupCpuQuota,
    ThermalThrottle,
    ScalingGovernor,
    MajorFaults,
    CompetingLoad,
]


def sample_all(root: str = "/") -> Dict[str, Reading]:
    return {p.name: p(root).sample() for p in ALL_PROBES}
