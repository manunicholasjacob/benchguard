"""benchguard: decide whether a performance measurement was valid, and say why if it was not.

    from benchguard import guard

    with guard(threads=4) as g:
        result = run_benchmark()

    if not g.verdict.ok():
        raise SystemExit(g.verdict.summary())

The checks are not a wish list. Each one exists because its absence produced a wrong published
number somewhere, and the magnitude in each docstring is the measured cost of ignoring it.
"""
from .probes import ALL_PROBES, Reading, sample_all
from .verdict import FAIL, PASS, UNKNOWN, Finding, Verdict, evaluate, guard

__version__ = "0.1.0"

__all__ = [
    "guard", "evaluate", "sample_all", "Verdict", "Finding", "Reading",
    "ALL_PROBES", "PASS", "FAIL", "UNKNOWN", "__version__",
]
