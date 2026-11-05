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


class CgroupProbe(Probe):
    """Shared base: find the cgroup this process is actually in.

    Reading `/sys/fs/cgroup/cpu.stat` directly reads the ROOT cgroup, which is not where a
    throttled process usually lives. Under `systemd-run --scope`, inside a container, or under
    any CI runner that nests its jobs, the quota that binds is on a descendant, and the root's
    counters stay at zero while the job is being throttled hard. Found by trying to write a test
    that throttles something on purpose.

    cgroup v2 reports the path in `/proc/self/cgroup` as `0::/relative/path`.
    """

    def _cgroup_dir(self) -> str:
        base = "sys/fs/cgroup"
        text = self._read("proc/self/cgroup")
        if not text:
            return base
        for line in text.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                rel = parts[2].strip()
                if rel and rel != "/":
                    candidate = os.path.join(base, rel.lstrip("/"))
                    if os.path.isdir(self._path(candidate)):
                        return candidate
                break
        return base

    def _chain(self) -> List[str]:
        """This cgroup and every ancestor up to the root, nearest first.

        A quota binds at whichever level is tightest, so the effective limit is the minimum
        along this chain, not whatever the leaf happens to say.
        """
        base = "sys/fs/cgroup"
        here = self._cgroup_dir()
        out = [here]
        while here != base and here.startswith(base):
            here = os.path.dirname(here)
            out.append(here)
        return out


class CgroupCpuThrottle(CgroupProbe):
    """cgroup v2 `cpu.stat`: how long the kernel refused to schedule us.

    This is the check with the largest measured magnitude behind it. A CPU quota that binds
    costs roughly half of throughput on Cortex-A76 and roughly two thirds on x86, and the
    throttle counters rise monotonically with the loss, so a non-zero delta here is not a
    warning about a possible effect, it is the effect.
    """

    name = "cgroup_cpu_throttle"

    def sample(self) -> Reading:
        cgroup = self._cgroup_dir()
        text = self._read(cgroup, "cpu.stat")
        if text is None and cgroup != "sys/fs/cgroup":
            text = self._read("sys/fs/cgroup/cpu.stat")
        if text is None:
            return Reading.missing(self.name, "no cpu.stat (not cgroup v2, or not Linux)")
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


class CgroupCpuQuota(CgroupProbe):
    """cgroup v2 `cpu.max`: the quota itself, in cores.

    Static, so it is read once rather than differenced. Reported because a quota smaller than
    the thread count is the configuration that produces the throttling above, and knowing the
    ratio turns "it was throttled" into "it was throttled because you asked for more threads
    than cores".
    """

    name = "cgroup_cpu_quota"

    def sample(self) -> Reading:
        seen = False
        tightest: Optional[float] = None
        raw_of_tightest = ""
        for d in self._chain():
            text = self._read(d, "cpu.max")
            if text is None:
                continue
            seen = True
            parts = text.split()
            if not parts or parts[0] == "max":
                continue                      # unlimited at this level, keep walking up
            try:
                quota, period = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                return Reading.missing(self.name, "cpu.max unparseable: %r" % text.strip())
            if period <= 0:
                return Reading.missing(self.name, "cpu.max period is zero")
            cores = quota / period
            if tightest is None or cores < tightest:
                tightest, raw_of_tightest = cores, text.strip()
        if not seen:
            return Reading.missing(self.name, "no cpu.max in this cgroup or its ancestors")
        if tightest is None:
            return Reading(name=self.name, value=None, unit="cores", raw="max",
                           extra={"unlimited": 1.0})
        return Reading(name=self.name, value=tightest, unit="cores", raw=raw_of_tightest,
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


class PiThrottle(Probe):
    """Raspberry Pi throttling, which is the one place the sysfs counters are not.

    A Pi 5 has no `thermal_throttle` directory at all. Verified on hardware rather than assumed:
    `/sys/devices/system/cpu/cpu0/thermal_throttle` does not exist on a Pi 5 running
    6.12.93+rpt-rpi-2712. So `ThermalThrottle` reports "not checked" there, and a Pi is exactly
    the machine where thermal throttling is most likely, which makes that the worst possible
    place to have a blind spot.

    The firmware answers `vcgencmd get_throttled` with a bitmask. The low bits are live state and
    the high bits are sticky since boot:

        0  under-voltage detected          16  under-voltage has occurred
        1  Arm frequency capped            17  Arm frequency capping has occurred
        2  currently throttled             18  throttling has occurred
        3  soft temperature limit active   19  soft temperature limit has occurred

    The sticky bits are set-only, so the number of them set never decreases and a rise across the
    measurement means a new condition appeared while the benchmark was running. That is the only
    part that can be attributed to the run. The live bits are reported too, because a measurement
    that began with the board already capped is worth knowing about even though it is not a delta.

    Under-voltage earns the same treatment as heat. On a Pi an undersized supply caps the clock
    just as a hot die does, the firmware reports both through this one word, and nothing in sysfs
    shows either.

    This is the only probe that runs a subprocess instead of reading a file, so it takes an
    injectable runner. The fake-machine fixtures cannot cover it on their own, which is the
    mistake that let the child-process accounting bug through, so there is also a test that runs
    against real firmware and skips everywhere else.
    """

    name = "pi_throttle"

    STICKY = {
        16: "under-voltage has occurred",
        17: "Arm frequency capping has occurred",
        18: "throttling has occurred",
        19: "soft temperature limit has occurred",
    }
    LIVE = {
        0: "under-voltage detected",
        1: "Arm frequency capped",
        2: "currently throttled",
        3: "soft temperature limit active",
    }

    def __init__(self, root: str = "/", runner=None):
        super().__init__(root)
        if runner is not None:
            self._runner = runner
        elif self.root in ("/", ""):
            self._runner = self._vcgencmd
        else:
            # A relocated root means the caller is describing some other machine, so running
            # THIS machine's firmware would be a straightforward lie. It reads a captured
            # answer from the tree instead, or reports that it could not look. Without this the
            # suite would pass on a laptop and quietly change behaviour on a real Pi, where
            # every fake machine would be answered by the firmware in the room.
            self._runner = self._from_root

    def _from_root(self) -> Optional[str]:
        return self._read("run/benchguard/vcgencmd_get_throttled")

    @staticmethod
    def _vcgencmd() -> Optional[str]:
        import shutil
        import subprocess
        exe = shutil.which("vcgencmd")
        if exe is None:
            return None
        try:
            done = subprocess.run([exe, "get_throttled"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return done.stdout.decode("utf-8", "replace").strip()

    def sample(self) -> Reading:
        text = self._runner()
        if not text:
            return Reading.missing(
                self.name,
                "vcgencmd get_throttled not available (not a Raspberry Pi, or vcgencmd missing)")
        m = re.search(r"throttled=0x([0-9a-fA-F]+)", text)
        if m is None:
            return Reading.missing(
                self.name, "could not parse vcgencmd output %r" % text[:60])
        bits = int(m.group(1), 16)
        sticky = [d for b, d in sorted(self.STICKY.items()) if bits & (1 << b)]
        live = [d for b, d in sorted(self.LIVE.items()) if bits & (1 << b)]
        extra = {"bits": float(bits), "live_count": float(len(live))}
        for b in self.STICKY:
            extra["sticky_%d" % b] = 1.0 if bits & (1 << b) else 0.0
        return Reading(name=self.name, value=float(len(sticky)), unit="conditions",
                       raw="0x%x" % bits, extra=extra)


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
    """Major page faults taken by this process and by the children it reaped.

    Major faults mean the measurement waited on storage. On a 272,000-page model a run that
    took 11,404 of them is not measuring the model, it is measuring the page cache, and the
    difference does not show up in wall-clock variance in any way an eye can catch.

    majflt (field 12) covers this process; cmajflt (field 13) covers children it has waited
    for. `benchguard run` does all of its work in a child, so without cmajflt the headline use
    of this package would report "no major page faults" however hard the benchmark paged.
    cmajflt only lands when the child is reaped, which the CLI does before it samples. A
    benchmark that leaves a grandchild running is outside what this can see.
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
        # after comm and state, fields[k] is stat field 3+k: majflt (12) is fields[9] and
        # cmajflt (13) is fields[10]
        try:
            majflt = float(fields[9])
            cmajflt = float(fields[10])
        except (IndexError, ValueError):
            return Reading.missing(self.name, "/proc/self/stat too short for majflt")
        return Reading(name=self.name, value=majflt + cmajflt, unit="faults",
                       extra={"self": majflt, "children": cmajflt})


class CompetingLoad(Probe):
    """Busy jiffies on the machine that were not spent on this process.

    This is the check that comes from the most expensive lesson in the set. A phone left idle
    and charging ran background application compilation during the measurement, cost 32% of a
    memory-bandwidth ceiling, and every scatter-based health check on the run passed. Load
    average is useless for this because it is a decayed average; the process table is not.

    "This process" has to mean the process tree, not the current process. utime and stime
    (fields 14, 15) are this process alone, and `benchguard run` spends almost no CPU itself:
    the benchmark is a child. Judged on utime and stime alone, every real CLI run reported
    that ~100% of the machine's busy time belonged to other processes and rejected a perfectly
    good measurement. The first CI run on a quiet hosted runner failed exactly that way, which
    is the argument for dogfooding the Action rather than trusting the unit tests. cutime and
    cstime (fields 16, 17) carry the reaped children, so the sum covers both the CLI, where
    the work is a child, and the plugin and context manager, where it is in-process.
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
            self_busy = float(fields[11]) + float(fields[12])    # utime + stime
            child_busy = float(fields[13]) + float(fields[14])   # cutime + cstime
        except (IndexError, ValueError):
            return Reading.missing(self.name, "/proc/self/stat too short for utime/stime")
        own_busy = self_busy + child_busy
        return Reading(name=self.name, value=busy, unit="jiffies",
                       extra={"own_busy": own_busy, "idle": idle,
                              "self_busy": self_busy, "child_busy": child_busy})


ALL_PROBES = [
    CgroupCpuThrottle,
    CgroupCpuQuota,
    ThermalThrottle,
    PiThrottle,
    ScalingGovernor,
    MajorFaults,
    CompetingLoad,
]


def sample_all(root: str = "/") -> Dict[str, Reading]:
    return {p.name: p(root).sample() for p in ALL_PROBES}
