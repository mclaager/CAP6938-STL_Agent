import pytest

from stl_agent.config import clear_overrides, reload_settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Built-in defaults only: a fresh settings file in a temp folder, and no --set overrides."""
    monkeypatch.setenv("CAD_SETTINGS", str(tmp_path / "settings.toml"))
    clear_overrides()
    reload_settings()
    yield
    monkeypatch.setenv("CAD_SETTINGS", str(tmp_path / "settings-after.toml"))  # a test may have broken the file
    clear_overrides()
    reload_settings()
