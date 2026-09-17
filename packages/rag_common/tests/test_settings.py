from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import ValidationError

from rag_common.settings import BaseServiceSettings


def test_env_aliases_and_csv_parsing() -> None:
    s = BaseServiceSettings()
    assert s.service_name == "test-service"
    assert s.allowed_hosts == ["testserver", "localhost", "127.0.0.1"]
    assert s.cors_allow_origins == ["http://localhost:5678"]
    assert s.rate_limit_requests == 5
    assert s.otel_exporter_otlp_endpoint is None  # "" -> None
    assert s.service_jwt_secret.get_secret_value().startswith("unit-test-secret")
    assert "unit-test-secret" not in repr(s)  # SecretStr never leaks into logs


def test_weak_jwt_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_JWT_SECRET", "short")
    with pytest.raises(ValidationError):
        BaseServiceSettings()


def test_lru_cache_pattern_reads_env_once(monkeypatch: pytest.MonkeyPatch) -> None:
    @lru_cache
    def get_settings() -> BaseServiceSettings:
        return BaseServiceSettings()

    first = get_settings()
    monkeypatch.setenv("SERVICE_NAME", "changed")
    assert get_settings() is first
    assert get_settings().service_name == "test-service"
