"""Unit tests for pipeline.config."""

import pytest

from pipeline import config


def test_validate_settings_success(monkeypatch):
    """validate_settings passes with valid config."""
    config.validate_settings()


def test_validate_settings_invalid_port(monkeypatch):
    """validate_settings raises when QDRANT_PORT is invalid."""
    monkeypatch.setattr(config.settings, "QDRANT_PORT", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "QDRANT_PORT" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "QDRANT_PORT", 6333)


def test_validate_settings_invalid_model_name(monkeypatch):
    """validate_settings raises when MODEL_NAME is invalid."""
    orig = config.settings.MODEL_NAME
    monkeypatch.setattr(config.settings, "MODEL_NAME", "invalid")
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "MODEL_NAME" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "MODEL_NAME", orig)


def test_validate_settings_negative_upload_bytes(monkeypatch):
    """validate_settings raises when MAX_UPLOAD_BYTES is negative."""
    orig = config.settings.MAX_UPLOAD_BYTES
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_BYTES", -1)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "MAX_UPLOAD_BYTES" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_BYTES", orig)


def test_validate_settings_invalid_ffmpeg_timeout(monkeypatch):
    """validate_settings raises when FFMPEG_TIMEOUT_SEC < 1."""
    orig = config.settings.FFMPEG_TIMEOUT_SEC
    monkeypatch.setattr(config.settings, "FFMPEG_TIMEOUT_SEC", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "FFMPEG_TIMEOUT_SEC" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "FFMPEG_TIMEOUT_SEC", orig)


def test_parse_allowed_paths_empty():
    """_parse_allowed_paths returns [] for None or empty string."""
    assert config._parse_allowed_paths(None) == []
    assert config._parse_allowed_paths("") == []
    assert config._parse_allowed_paths("   ") == []


def test_parse_allowed_paths_single():
    """_parse_allowed_paths returns single path."""
    assert config._parse_allowed_paths("/tmp") == ["/tmp"]


def test_parse_allowed_paths_multiple():
    """_parse_allowed_paths returns comma-separated paths."""
    result = config._parse_allowed_paths("/tmp, /var, /home")
    assert result == ["/tmp", "/var", "/home"]
