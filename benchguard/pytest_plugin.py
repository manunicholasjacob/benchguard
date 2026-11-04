"""pytest plugin: fail a benchmark test whose machine state invalidated it.

    def test_throughput(benchguard):
        with benchguard(threads=4):
            result = run()
        assert result.ops_per_sec > 1000

The fixture is a context manager factory rather than a bare fixture, because the state that
matters is the state during the measurement, and a fixture that samples at setup time would be
measuring the wrong window.

A test that is otherwise green but ran on a throttled machine fails here, with the counter that
moved in the message. That is the point: a benchmark assertion passing on a contaminated run is
worse than it failing, because it gets believed.

Enable with `-p benchguard.pytest_plugin`, or install the package, which registers it.
"""
from __future__ import annotations

import pytest

from .verdict import guard as _guard


def pytest_addoption(parser):
    group = parser.getgroup("benchguard")
    group.addoption("--benchguard-strict", action="store_true", default=False,
                    help="fail when a validity check could not be made, not only when one fails")
    group.addoption("--benchguard-root", default="/",
                    help="filesystem root to read machine state from")
    group.addoption("--benchguard-competing-fraction", type=float, default=0.10,
                    help="share of machine busy time allowed to other processes")
    group.addoption("--benchguard-warn-only", action="store_true", default=False,
                    help="report invalid measurements as warnings instead of failures")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchguard(threads=None): judge the machine state around this test's measurement")


class _Session:
    """What the fixture hands back. Holds the verdict after the block exits."""

    def __init__(self, config, request):
        self._config = config
        self._request = request
        self.verdict = None

    def __call__(self, threads=None, competing_fraction=None):
        cfg = self._config
        self._cm = _guard(
            root=cfg.getoption("--benchguard-root"),
            threads=threads,
            competing_fraction=(cfg.getoption("--benchguard-competing-fraction")
                                if competing_fraction is None else competing_fraction),
        )
        return self

    def __enter__(self):
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)
        self.verdict = self._cm.verdict
        if exc_type is not None:
            return False          # the test's own failure wins
        self._judge()
        return False

    def _judge(self):
        cfg = self._config
        strict = cfg.getoption("--benchguard-strict")
        if self.verdict is None or self.verdict.ok(strict=strict):
            return
        message = "benchguard: " + self.verdict.summary(strict=strict)
        if cfg.getoption("--benchguard-warn-only"):
            import warnings
            warnings.warn(message, stacklevel=2)
        else:
            pytest.fail(message, pytrace=False)


@pytest.fixture
def benchguard(request):
    """Context-manager factory that judges the machine around the measurement."""
    return _Session(request.config, request)
