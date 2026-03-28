"""Shared pytest configuration and custom marks."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA GPU (skip with -m 'not gpu')",
    )
