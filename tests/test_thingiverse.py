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


def search_api(hits, files=None, licenses=None, seen=None, details=None):
    """Mock API for search: the hits, each thing's files (default: one STL) and details (its license, plus
    `details[thing]`, e.g. tags)."""
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
                                         "allows_derivatives": thing != 3, **(details or {}).get(thing, {})})
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


# ---------------------------------------------------------------------------
# Is a result the requested object? (#39)
# ---------------------------------------------------------------------------
def test_object_words_skip_kinds_of_model_and_sizes():
    assert tv.object_words("tree", ["low-poly tree", "tree"]) == {"tree"}
    assert tv.object_words("horse-statue", "horse statue") == {"horse"}
    assert tv.object_words("cat", ["sitting cat"]) == {"cat"}
    assert tv.object_words("flower-60mm", "tulip flower") == {"tulip"}  # nothing left in the name: the phrase
    assert tv.object_words("", "apple fruit") == {"apple"}
    assert tv.object_words("", "figurine") == {"figurine"}  # only generic words: still something to look for


def cand(name, tags=()):
    return tv.Candidate(id=1, name=name, url="u", thumbnail="", author="a", license="l", tags=list(tags))


def test_match_checks_name_then_tags_in_singular_and_plural():
    assert tv.match(cand("Low Poly Tree"), {"tree"}) == tv.NAME_MATCH
    assert tv.match(cand("Oak Trees set"), {"tree"}) == tv.NAME_MATCH
    assert tv.match(cand("Butterflies"), {"butterfly"}) == tv.NAME_MATCH
    assert tv.match(cand("Foxes"), {"fox"}) == tv.NAME_MATCH
    assert tv.match(cand("Low-Poly Bulbasaur", ["bulbasaur", "low-poly", "nintendo"]), {"tree"}) == 0
    assert tv.match(cand("Nature scene", ["forest", "pine tree"]), {"tree"}) == tv.TAG_MATCH
    assert tv.match(cand("Street lamp"), {"tree"}) == 0  # whole words only


# The search from the issue: "low-poly tree" ranks low-poly Pokémon above trees.
POKEMON = [hit(1, name="Low-Poly Bulbasaur"), hit(2, name="Low-Poly Charmander"), hit(3, name="Low poly forest"),
           hit(4, name="Low poly tree sculptures"), hit(5, name="Low-Poly Squirtle")]
POKEMON_TAGS = {1: {"tags": [{"name": "bulbasaur"}, {"name": "low-poly"}]}, 3: {"tags": [{"name": "tree"}]}}


def test_strict_search_keeps_only_the_object_name_matches_first():
    seen = []
    found = tv.search("low-poly tree", n=5, words={"tree"}, strict=True,
                      client=search_api(POKEMON, details=POKEMON_TAGS, seen=seen))
    assert [(c.id, c.match) for c in found] == [(4, tv.NAME_MATCH), (3, tv.TAG_MATCH)]
    assert found[1].tags == ["tree"]
    assert seen[0].url.params["per_page"] == str(tv.POOL)  # a larger pool: few hits match
    files = [r for r in seen if r.url.path.endswith("/files")]
    assert [r.url.path for r in files] == ["/things/3/files", "/things/4/files"]  # non-matches cost one request


def test_ranked_search_keeps_everything_but_lists_matches_first():
    found = tv.search("low-poly tree", n=5, words={"tree"}, client=search_api(POKEMON, details=POKEMON_TAGS))
    assert [c.id for c in found] == [4, 3, 1, 2, 5]
    assert [c.id for c in tv.search("low-poly tree", n=5, client=search_api(POKEMON))] == [1, 2, 3, 4, 5]


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


def test_thumbnails_are_fetched_with_our_user_agent():
    """The app shows fetched thumbnails: the image server refuses some clients (#9 gallery tiles were blank)."""
    agents = []

    def handler(request):
        agents.append(request.headers.get("user-agent"))
        if request.url.path == "/ok.png":
            return httpx.Response(200, content=b"PNG", headers={"content-type": "image/png"})
        return httpx.Response(403, content=b"<html>challenge</html>", headers={"content-type": "text/html"})

    with httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": tv.USER_AGENT}) as c:
        got = tv.thumbnails(["https://img.example/ok.png", "https://img.example/blocked.png", ""], client=c)
    assert got == [b"PNG", None, None] and agents == [tv.USER_AGENT] * 2
