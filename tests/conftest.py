"""Shared test configuration.  (2026-10-14)

chladni is douse's backend; lappy is a deprecated optional extra (`douse[lappy]`). Tests of the
lappy backend's internals -- its tension thresholds, Weyl enforcement, tracking verification --
are marked `lappy` and skip when lappy is not installed.
"""
import importlib.util
import inspect
import os

import pytest


def _lappy_works():
    """lappy importable AND able to run: it calls `gsvd4py.gsvdvals(tola=..., tolb=...)`, which
    gsvd4py 0.4 (chladni's requirement) no longer accepts, so an installed lappy can still be
    unusable beside chladni. `DOUSE_NO_LAPPY=1` forces the skip."""
    if os.environ.get("DOUSE_NO_LAPPY"):
        return False
    if importlib.util.find_spec("lappy") is None:
        return False
    try:
        import gsvd4py
        return "tola" in inspect.signature(gsvd4py.gsvdvals).parameters
    except Exception:
        return False


HAVE_LAPPY = _lappy_works()


def pytest_configure(config):
    config.addinivalue_line("markers", "lappy: tests the deprecated lappy backend (douse[lappy])")
    config.addinivalue_line("markers", "slow: a long-running test")


def pytest_collection_modifyitems(config, items):
    if HAVE_LAPPY:
        return
    skip = pytest.mark.skip(reason="needs a working lappy, the deprecated backend (douse[lappy] "
                            "with a gsvd4py that still takes tola/tolb)")
    for item in items:
        if "lappy" in item.keywords:
            item.add_marker(skip)
