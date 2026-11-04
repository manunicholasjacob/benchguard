"""End-to-end tests for the command line, including the exit codes CI depends on.

The important thing these cover that the unit tests cannot: the delta path. benchguard samples
the machine before and after the command, so a fixture that does not change between those two
samples exercises nothing. Each test here therefore runs a command that *mutates the fake
machine while it runs*, which is what a real throttling event does.

Getting this wrong is easy and quiet. An earlier version of this test used a static fixture, saw
the throttled machine exit 0, and would have been read as the gate failing to fire when in fact
the test was failing to throttle anything.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from test_benchguard import make_machine

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
TESTS = str(__import__("pathlib").Path(__file__).resolve().parent)

EXIT_INVALID = 90
EXIT_NOT_PROVEN = 91


def mutating_command(root, **after_state):
    """A command that rewrites the fake machine as it runs, then does a little work."""
    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from test_benchguard import make_machine\n"
        "make_machine(%r, **%r)\n"
        "sum(range(200000))\n" % (TESTS, str(root), after_state)
    )
    return [sys.executable, "-c", script]


def run_cli(root, command, *extra):
    argv = ([sys.executable, "-m", "benchguard.cli", "run", "--root", str(root),
             "--threads", "2"] + list(extra) + ["--"] + command)
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO)


@pytest.fixture
def machine(tmp_path):
    root = tmp_path / "m"
    make_machine(str(root), thermal=0, cpu_busy=1000.0, own_busy=1000.0)
    return root


def test_clean_run_exits_zero(machine):
    r = run_cli(machine, mutating_command(machine, thermal=0, cpu_busy=2000.0, own_busy=2000.0))
    assert r.returncode == 0, r.stderr
    assert "valid" in r.stderr


def test_throttling_during_the_run_is_caught(machine):
    r = run_cli(machine, mutating_command(machine, thermal=0, throttled_usec=400000,
                                          nr_throttled=9, cpu_busy=2000.0, own_busy=2000.0))
    assert r.returncode == EXIT_INVALID, r.stderr
    assert "throttled" in r.stderr.lower()


def test_competing_load_during_the_run_is_caught(machine):
    # 320 of 1000 new busy jiffies belong to someone else
    r = run_cli(machine, mutating_command(machine, thermal=0, cpu_busy=2000.0, own_busy=1680.0))
    assert r.returncode == EXIT_INVALID, r.stderr
    assert "other processes" in r.stderr


def test_major_faults_during_the_run_are_caught(machine):
    r = run_cli(machine, mutating_command(machine, thermal=0, majflt=11404,
                                          cpu_busy=2000.0, own_busy=2000.0))
    assert r.returncode == EXIT_INVALID, r.stderr
    assert "major page faults" in r.stderr


def test_the_commands_own_failure_outranks_the_verdict(machine):
    """A broken benchmark is a different problem from an untrustworthy one, and must not be
    reported as the latter."""
    r = run_cli(machine, [sys.executable, "-c", "raise SystemExit(3)"])
    assert r.returncode == 3, r.stderr


def test_strict_mode_refuses_a_machine_it_cannot_inspect(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    r = run_cli(empty, [sys.executable, "-c", "pass"], "--strict")
    assert r.returncode == EXIT_NOT_PROVEN, r.stderr
    assert "NOT PROVEN" in r.stderr


def test_without_strict_an_uninspectable_machine_is_not_a_failure(tmp_path):
    empty = tmp_path / "nothing2"
    empty.mkdir()
    r = run_cli(empty, [sys.executable, "-c", "pass"])
    assert r.returncode == 0, r.stderr


def test_provenance_record_is_written_and_parses(machine, tmp_path):
    out = tmp_path / "record.json"
    r = run_cli(machine, mutating_command(machine, thermal=0, cpu_busy=2000.0, own_busy=2000.0),
                "--json", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["provenance"]["threads_declared"] == 2
    assert doc["provenance"]["platform"]
    assert doc["duration_s"] > 0
    assert {f["check"] for f in doc["findings"]} >= {"cgroup_cpu_throttle", "competing_load"}


def test_sample_subcommand_accepts_flags_after_the_subcommand(machine):
    """argparse puts parent options before the subcommand by default, which is not how anyone
    types it."""
    r = subprocess.run([sys.executable, "-m", "benchguard.cli", "sample", "--root", str(machine)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "cgroup_cpu_throttle" in r.stdout


def test_nothing_to_run_is_a_usage_error(machine):
    r = subprocess.run([sys.executable, "-m", "benchguard.cli", "run", "--root", str(machine)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2
