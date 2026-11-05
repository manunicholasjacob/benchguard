"""Raspberry Pi firmware throttling, which is the one machine sysfs cannot answer for.

A Pi 5 has no `/sys/devices/system/cpu/cpu0/thermal_throttle` directory at all. That is not an
inference from documentation: it was checked on a Pi 5 Model B Rev 1.0 running
`6.12.93+rpt-rpi-2712`, where the path does not exist. So the probe that every other machine
uses for heat returns "not checked" on the board most likely to be throttled by heat, and the
firmware's own `vcgencmd get_throttled` is the only source.

These tests also pin the design decision that came out of writing them. Reporting sysfs and the
Pi firmware as two separate checks meant strict mode could not be satisfied on any machine at
all, because no machine has both sources: an x86 box is missing vcgencmd and a Pi is missing the
counters. `test_strict_mode_is_satisfiable_on_a_pi` is the regression test for that.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from benchguard.probes import PiThrottle, sample_all
from benchguard.verdict import FAIL, PASS, UNKNOWN, evaluate

from test_benchguard import make_machine

# bit 18 is "throttling has occurred", bit 2 is "currently throttled",
# bit 16 is "under-voltage has occurred", bit 0 is "under-voltage detected"
OCCURRED = 1 << 18
LIVE_THROTTLE = (1 << 2) | OCCURRED
UNDERVOLT_OCCURRED = 1 << 16


def finding(verdict, check):
    return next(f for f in verdict.findings if f.check == check)


def around(root, before_kw, after_kw):
    make_machine(root, **before_kw)
    b = sample_all(root)
    make_machine(root, **after_kw)
    return evaluate(b, sample_all(root), duration_s=1.0)


PI = dict(thermal=None, pi_throttled=0, cpu_busy=1000.0, own_busy=1000.0)


# ------------------------------------------------------------------ the probe itself ----

@pytest.mark.parametrize("word,sticky,live", [
    (0x0, 0, 0),
    (OCCURRED, 1, 0),
    (LIVE_THROTTLE, 1, 1),
    (0x50005, 2, 2),          # under-voltage and throttling, both live and both recorded
    (UNDERVOLT_OCCURRED, 1, 0),
])
def test_the_bitmask_is_decoded_as_the_firmware_documents_it(word, sticky, live):
    r = PiThrottle("/", runner=lambda: "throttled=0x%x" % word).sample()
    assert r.available
    assert r.value == sticky, r.raw
    assert r.extra["live_count"] == live
    assert r.raw == "0x%x" % word


def test_an_absent_vcgencmd_is_reported_as_unchecked_not_as_clean():
    r = PiThrottle("/", runner=lambda: None).sample()
    assert not r.available
    assert "vcgencmd" in r.why_unavailable


def test_unparseable_output_is_also_unchecked():
    r = PiThrottle("/", runner=lambda: "bcm2712: everything is fine").sample()
    assert not r.available
    assert "parse" in r.why_unavailable


def test_a_relocated_root_never_runs_this_machines_firmware(tmp_path, monkeypatch):
    """The important one, and the reason the probe does not simply shell out.

    Every other probe reads from a relocatable root, which is what lets the suite describe a
    Linux machine from a Windows laptop. A probe that shells out would answer every fake machine
    with the firmware of the machine in the room, so the suite would pass here and silently
    change behaviour when run on a real Pi.
    """
    called = []
    monkeypatch.setattr(shutil, "which", lambda name: called.append(name) or "/usr/bin/vcgencmd")
    r = PiThrottle(str(tmp_path)).sample()
    assert called == [], "a relocated root must not consult the local firmware"
    assert not r.available


def test_a_relocated_root_reads_a_captured_answer_when_the_tree_has_one(tmp_path):
    root = str(tmp_path / "pi")
    make_machine(root, **dict(PI, pi_throttled=OCCURRED))
    r = sample_all(root)["pi_throttle"]
    assert r.available and r.value == 1.0


# --------------------------------------------------------------------- the verdict ------

def test_a_clean_pi_passes_and_says_which_source_answered(tmp_path):
    v = around(str(tmp_path / "pi"), dict(PI), dict(PI, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == PASS
    assert "get_throttled" in f.detail
    assert v.ok()


def test_throttling_that_begins_during_the_measurement_fails(tmp_path):
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=0),
               dict(PI, pi_throttled=OCCURRED, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == FAIL
    assert not v.ok()


def test_still_being_throttled_at_the_end_fails_even_with_no_new_sticky_bit(tmp_path):
    """The board was already capped before the run and stayed capped through it.

    No sticky bit changes, so a delta alone would call this clean. The number is still junk.
    """
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=LIVE_THROTTLE),
               dict(PI, pi_throttled=LIVE_THROTTLE, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == FAIL
    assert "still capped" in f.detail
    # and it says which condition, not just that there was one
    assert "currently throttled" in f.detail


def test_throttling_that_happened_earlier_today_does_not_fail_this_measurement(tmp_path):
    """Sticky bits persist until reboot, so treating them as a failure would condemn every
    measurement taken for the rest of the uptime. It is reported, not failed."""
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=OCCURRED),
               dict(PI, pi_throttled=OCCURRED, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == PASS
    assert "earlier since boot" in f.detail
    assert v.ok()


def test_under_voltage_is_treated_the_same_as_heat_and_is_named_as_such(tmp_path):
    """On a Pi an undersized supply caps the clock exactly as a hot die does, and the firmware
    reports both through this one word.

    It has to say which, though. "2 conditions" sends nobody anywhere; under-voltage means buy
    a better supply and heat means improve the cooling.
    """
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=0),
               dict(PI, pi_throttled=UNDERVOLT_OCCURRED, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == FAIL
    assert "under-voltage has occurred" in f.detail
    assert "throttling has occurred" not in f.detail, "named a condition that did not happen"


def test_two_conditions_arising_at_once_are_both_named(tmp_path):
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=0),
               dict(PI, pi_throttled=UNDERVOLT_OCCURRED | OCCURRED,
                    cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == FAIL
    assert "under-voltage has occurred" in f.detail
    assert "throttling has occurred" in f.detail


def test_a_condition_already_set_before_the_run_is_not_named_as_new(tmp_path):
    """The delta is what gets attributed to the run, so a bit that was already set must not be
    reported as something this measurement caused."""
    v = around(str(tmp_path / "pi"),
               dict(PI, pi_throttled=UNDERVOLT_OCCURRED),
               dict(PI, pi_throttled=UNDERVOLT_OCCURRED | OCCURRED,
                    cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == FAIL
    assert "throttling has occurred" in f.detail
    assert "under-voltage" not in f.detail, "blamed the run for a pre-existing condition"


def test_sysfs_wins_when_a_machine_somehow_has_both(tmp_path):
    v = around(str(tmp_path / "x86"),
               dict(thermal=0, pi_throttled=OCCURRED, cpu_busy=1000.0, own_busy=1000.0),
               dict(thermal=0, pi_throttled=OCCURRED, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == PASS
    assert "get_throttled" not in f.detail


def test_neither_source_is_unknown_and_names_both_reasons(tmp_path):
    v = around(str(tmp_path / "bare"),
               dict(thermal=None, cpu_busy=1000.0, own_busy=1000.0),
               dict(thermal=None, cpu_busy=2000.0, own_busy=2000.0))
    f = finding(v, "thermal_throttle")
    assert f.outcome == UNKNOWN
    assert "thermal_throttle counters" in f.detail and "vcgencmd" in f.detail


def test_strict_mode_is_satisfiable_on_a_pi(tmp_path):
    """The regression test for the design mistake. While sysfs and the firmware were two
    separate checks, one of them was unavailable on every machine that exists, so strict mode
    failed on a perfectly well described Pi and on a perfectly well described x86 box."""
    v = around(str(tmp_path / "pi"), dict(PI), dict(PI, cpu_busy=2000.0, own_busy=2000.0))
    assert v.ok(strict=True), v.summary(strict=True)


# ------------------------------------------------------------------- real firmware ------

@pytest.mark.skipif(shutil.which("vcgencmd") is None,
                    reason="no vcgencmd, so this is not a Raspberry Pi")
def test_against_the_real_firmware():
    """Fixtures write whatever the probe expects to read, which is exactly how the child-process
    accounting bug survived a full suite. This runs the real binary on real hardware."""
    out = subprocess.run(["vcgencmd", "get_throttled"], stdout=subprocess.PIPE)
    assert out.returncode == 0
    text = out.stdout.decode()
    assert text.startswith("throttled=0x"), text

    r = PiThrottle("/").sample()
    assert r.available, r.why_unavailable
    assert r.raw is not None and r.raw.startswith("0x")
    expected = int(text.split("=")[1], 16)
    assert r.extra["bits"] == float(expected)
    print("\nreal firmware says %s, decoded to %.0f recorded condition(s), %.0f live"
          % (text.strip(), r.value, r.extra["live_count"]), file=sys.stderr)
