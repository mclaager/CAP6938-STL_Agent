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


@pytest.fixture(autouse=True)
def no_thumbnail_downloads(monkeypatch):
    """The app fetches gallery thumbnails itself; in tests every fetch fails (no network), so links are shown."""
    import httpx

    from stl_agent.sources import thingiverse

    monkeypatch.setattr(thingiverse, "_thumbnail_client",
                        lambda: httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404))))
