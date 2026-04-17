"""Shared fixtures for integration tests."""
from __future__ import annotations

import pytest
from tests.support.db import PipelineMockConn, PipelineMockPool


@pytest.fixture
def mock_conn():
    return PipelineMockConn()


@pytest.fixture
def mock_pool(mock_conn):
    return PipelineMockPool(mock_conn)
