"""Shared test configuration.  (2026-10-14)

chladni is douse's backend; lappy is a deprecated optional extra (`douse[lappy]`). Tests of the
lappy backend's internals -- its tension thresholds, Weyl enforcement, tracking verification --
are marked `lappy` and skip when lappy is not installed.
"""
import importlib.util

import pytest

HAVE_LAPPY = importlib.util.find_spec("lappy") is not None


def pytest_configure(config):
    config.addinivalue_line("markers", "lappy: tests the deprecated lappy backend (douse[lappy])")
    config.addinivalue_line("markers", "slow: a long-running test")


def pytest_collection_modifyitems(config, items):
    if HAVE_LAPPY:
        return
    skip = pytest.mark.skip(reason="needs lappy, the deprecated backend: install douse[lappy]")
    for item in items:
        if "lappy" in item.keywords:
            item.add_marker(skip)
