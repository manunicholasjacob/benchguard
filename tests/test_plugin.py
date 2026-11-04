"""The pytest plugin, exercised by running pytest inside pytest.

A plugin that fails tests has to be tested from the outside, because the thing being asserted
is the outcome pytest reports, not a return value. These run a generated test file in a
subprocess and read the result.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from test_benchguard import make_machine

REPO = str(Path(__file__).resolve().parents[1])
TESTS = str(Path(__file__).resolve().parent)

TEMPLATE = '''
import sys
sys.path.insert(0, {tests!r})
from test_benchguard import make_machine

def test_measured(benchguard):
    with benchguard(threads=2):
        make_machine({root!r}, **{after!r})
    assert True
'''


def write_test(tmp_path, root, after):
    f = tmp_path / "test_generated.py"
    f.write_text(TEMPLATE.format(tests=TESTS, root=str(root), after=after), encoding="utf-8")
    return f


def run_pytest(testfile, root, *extra):
    """Run pytest the way a user would: installed, with the plugin auto-loaded.

    No `-p benchguard.pytest_plugin` here. The package declares a `pytest11` entry point, so
    once it is installed pytest loads the plugin itself, and passing `-p` as well registers it
    twice: `ValueError: Plugin already registered under a different name`. That passed locally
    only because the package was not installed there, and failed on every CI job that did the
    normal thing. The suite therefore requires `pip install -e .`.

    The file argument is a bare name and the subprocess runs in the file's own directory,
    rather than an absolute temp path while the cwd is the repository. Handed the absolute
    form, the Windows Python 3.9 CI job went off and collected the drive root instead,
    failing on `C:\\Documents and Settings` with WinError 5, while the Windows 3.13 job on
    identical code passed. The two jobs resolve different pytest versions (8.4.2 and 9.2.0),
    which is the obvious suspect, but that is not the cause: both versions take that exact
    argv happily on a local Windows box, checked directly. So the trigger is something about
    the runner that is not reproducible here, and the honest fix is to stop using the
    construct rather than to claim a diagnosis. Running in the file's own directory also makes
    rootdir the temp directory instead of leaving pytest to reconcile a cwd and an argument
    on two unrelated trees, which is the most likely trigger of the two.
    """
    testfile = Path(testfile)
    argv = [sys.executable, "-m", "pytest", testfile.name, "-q",
            "--benchguard-root", str(root)] + list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=str(testfile.parent))


@pytest.fixture
def machine(tmp_path):
    root = tmp_path / "m"
    make_machine(str(root), thermal=0, cpu_busy=1000.0, own_busy=1000.0)
    return root


def test_clean_measurement_passes(tmp_path, machine):
    f = write_test(tmp_path, machine, dict(thermal=0, cpu_busy=2000.0, own_busy=2000.0))
    r = run_pytest(f, machine)
    assert r.returncode == 0, r.stdout + r.stderr


def test_throttled_measurement_fails_the_test(tmp_path, machine):
    """The assertion in the test body is True. It still has to fail, because the number the
    assertion is about was taken on a throttled machine."""
    f = write_test(tmp_path, machine, dict(thermal=0, throttled_usec=400000, nr_throttled=9,
                                           cpu_busy=2000.0, own_busy=2000.0))
    r = run_pytest(f, machine)
    assert r.returncode != 0
    assert "benchguard" in r.stdout
    assert "throttled" in r.stdout.lower()


def test_warn_only_reports_without_failing(tmp_path, machine):
    f = write_test(tmp_path, machine, dict(thermal=0, throttled_usec=400000, nr_throttled=9,
                                           cpu_busy=2000.0, own_busy=2000.0))
    r = run_pytest(f, machine, "--benchguard-warn-only", "-W", "default")
    assert r.returncode == 0, r.stdout + r.stderr


def test_strict_mode_fails_on_an_uninspectable_machine(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    f = tmp_path / "test_plain.py"
    f.write_text("def test_x(benchguard):\n    with benchguard():\n        pass\n",
                 encoding="utf-8")
    assert run_pytest(f, empty).returncode == 0
    assert run_pytest(f, empty, "--benchguard-strict").returncode != 0
