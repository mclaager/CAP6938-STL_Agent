"""Thingiverse client against a mock HTTP transport (no network)."""

import json

import httpx
import pytest

from stl_agent.sources import thingiverse as tv
from stl_agent.sources.thingiverse import ThingiverseError

STL = b"solid x\nendsolid x\n"


def hit(i, **kw):
    return {"id": i, "name": f"Horse {i}", "public_url": f"https://www.thingiverse.com/thing:{i}",
            "thumbnail": f"https://cdn.example/{i}.jpg", "creator": {"name": f"maker{i}"},
            "license": "Creative Commons - Attribution", "allows_derivatives": True, "is_nsfw": False, **kw}


def file(i, name, size, **kw):
    return {"id": i, "name": name, "size": size, "download_url": f"{tv.API}/files/{i}/download",
            "direct_url": f"{tv.API}/v2/files/{i}/download?increment_download=false", **kw}


def client(handler):
    """A client like make_client's, on a mock transport; `handler(request)` returns an httpx.Response."""
    return httpx.Client(base_url=tv.API, transport=httpx.MockTransport(handler), follow_redirects=True,
                        headers={"Authorization": "Bearer t0ken", "User-Agent": tv.USER_AGENT})


def search_api(hits, files=None, licenses=None, seen=None):
    """Mock API for search: the hits, each thing's files (default: one STL) and details (its license)."""
    def handler(request):
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path.startswith("/search/"):
            return httpx.Response(200, json={"total": len(hits), "hits": hits})
        thing = int(path.split("/")[2])
        if path.endswith("/files"):
            return httpx.Response(200, json=(files or {}).get(thing, [file(thing, f"t{thing}.stl", 100)]))
        return httpx.Response(200, json={"id": thing, "license": (licenses or {}).get(thing, "CC-BY"),
                                         "allows_derivatives": thing != 3})
    return client(handler)


def test_search_keeps_things_with_stl_and_adds_license():
    seen = []
    hits = [{k: v for k, v in hit(i).items() if k != "license"} for i in (1, 3, 4, 5)]  # real hits have no license
    hits[1]["is_ai"] = True
    hits.insert(1, hit(2, is_nsfw=True))
    api = search_api(hits, files={4: [file(40, "horse.obj", 10)]}, licenses={1: "Creative Commons - Attribution"},
                     seen=seen)
    found = tv.search("horse  statue", n=5, client=api)
    assert [c.id for c in found] == [1, 3, 5]  # 2 is NSFW, 4 has only an .obj
    assert found[0].author == "maker1" and found[0].license == "Creative Commons - Attribution"
    assert found[0].allows_derivatives and not found[1].allows_derivatives
    assert found[1].is_ai and "AI" in found[1].caption()
    assert seen[0].url.raw_path.startswith(b"/search/horse%20statue/?")
    assert seen[0].url.params["type"] == "things"
    assert seen[0].headers["authorization"] == "Bearer t0ken"


def test_search_stops_at_setting_count():
    seen = []
    found = tv.search("horse", client=search_api([hit(i) for i in range(1, 20)], seen=seen))
    assert len(found) == 6  # thingiverse_results default
    assert len(seen) == 1 + 6 * 2  # no requests for hits past the sixth


def test_missing_token(monkeypatch):
    monkeypatch.delenv(tv.TOKEN_VAR, raising=False)
    assert not tv.has_token()
    with pytest.raises(ThingiverseError, match="THINGIVERSE_TOKEN"):
        tv.search("horse")


def test_rejected_token():
    with pytest.raises(ThingiverseError, match="rejected"):
        tv.search("horse", client=client(lambda r: httpx.Response(401, json={"error": "Unauthorized"})))


def test_make_client_sends_token_and_user_agent(monkeypatch):
    monkeypatch.setenv(tv.TOKEN_VAR, " t0ken ")
    with tv.make_client() as c:
        assert c.headers["authorization"] == "Bearer t0ken"
        assert c.headers["user-agent"] == tv.USER_AGENT  # Cloudflare refuses httpx's default


def test_download_takes_largest_stl_and_follows_cdn_redirect(tmp_path):
    files = [file(1, "base.stl", 100), file(2, "horse.STL", 5000), file(3, "horse.3mf", 9000),
             file(4, "ear.stl", 10)]
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/things/7/files":
            return httpx.Response(200, json=files)
        if request.url.host == "api.thingiverse.com":
            return httpx.Response(302, headers={"location": "https://cdn.thingiverse.com/assets/horse.STL"})
        return httpx.Response(200, content=STL, headers={"content-type": "application/sla"})

    path, others = tv.download_stl(7, tmp_path / "h.stl", client=client(handler))
    assert path.read_bytes() == STL
    assert others == ["base.stl", "ear.stl"]
    assert seen[1].url.path == "/files/2/download"  # download_url: counts the download for the author
    cdn = seen[2]
    assert cdn.url.host == "cdn.thingiverse.com"
    assert "authorization" not in cdn.headers  # the token only goes to the API
    assert cdn.headers["user-agent"] == tv.USER_AGENT


def test_cloudflare_block_is_explained():
    page = httpx.Response(403, text="<html>Just a moment...</html>", headers={"content-type": "text/html"})
    with pytest.raises(ThingiverseError, match="Cloudflare"):
        tv.search("horse", client=client(lambda r: page))


def test_no_stl_files(tmp_path):
    api = client(lambda r: httpx.Response(200, json=[file(1, "horse.3mf", 10), file(2, "big.stl", tv.MAX_BYTES + 1)]))
    with pytest.raises(ThingiverseError, match="no STL"):
        tv.download_stl(7, tmp_path / "h.stl", client=api)


def test_run_fetch_saves_attribution(monkeypatch, tmp_path):
    import trimesh

    from stl_agent import pipeline

    def fake_download(thing_id, dest, client=None):
        trimesh.creation.box((10, 20, 30)).export(dest)
        return dest, ["extra.stl"]

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "unique_stl_path", lambda name: tmp_path / f"{name}.stl")
    monkeypatch.setattr(tv, "download_stl", fake_download)
    c = tv._candidate(hit(9))
    log = []
    stl, report = pipeline.run_fetch(c, log.append, name="horse-statue")
    assert stl.endswith("horse-statue.stl")
    assert report["success"] and sorted(report["validation"]["bbox_mm"]) == [10, 20, 30]
    source = json.loads((tmp_path / "horse-statue.source.json").read_text())
    assert source["author"] == "maker9" and source["other_stl_files"] == ["extra.stl"]
    assert any("Saved" in line for line in log)
