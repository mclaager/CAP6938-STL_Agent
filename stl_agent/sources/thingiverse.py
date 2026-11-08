"""Thingiverse search and download (#9), for requests code generation can't build.

API: https://www.thingiverse.com/developers/swagger. Needs an app token (THINGIVERSE_TOKEN in .env, from
thingiverse.com/apps/create); without one, every request is refused. Generation works without it.

    search(term) -> [Candidate]                  only things with a downloadable STL, with their license
    search(term, words=..., strict=True)         ...whose name or tags hold one of the object words (#39)
    object_words(id_or_name, phrases) -> words   the object a piece or request asks for
    download_stl(thing_id, dest) -> (path, other STL names)

Cloudflare in front of the API and the file CDN answers 403 to default client User-Agents (curl's,
httpx's), token or not, so every request sends USER_AGENT.

Every function takes an optional `client` (an httpx.Client) so tests can pass one with a mock transport.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

from ..config import settings

API = "https://api.thingiverse.com"
TOKEN_VAR = "THINGIVERSE_TOKEN"
MAX_BYTES = 100 * 1024 * 1024  # skip files larger than this
TIMEOUT = 30
USER_AGENT = "stl-agent/0.1 (CAP 6938 course project)"  # any non-default value passes Cloudflare
POOL = 30  # hits looked at when results must match the object (#39): often only 1-2 of 10 do


class ThingiverseError(RuntimeError):
    pass


@dataclass
class Candidate:
    id: int
    name: str
    url: str  # the thing's page on thingiverse.com
    thumbnail: str
    author: str
    license: str
    allows_derivatives: bool = False  # matters when parts are combined (#10): the result is a derivative
    is_ai: bool = False
    tags: list[str] = field(default_factory=list)
    match: int = 0  # how it matched the object words (#39): NAME_MATCH, TAG_MATCH or 0 (not checked / none)

    def credit(self) -> str:
        """Attribution line: name, author, license and link."""
        return f"{self.name} by {self.author} ({self.license}), {self.url}"

    def caption(self) -> str:
        flags = (" [AI-generated]" if self.is_ai else "") + ("" if self.allows_derivatives else " [no derivatives]")
        return f"{self.name}, by {self.author} ({self.license}){flags}"

    def to_dict(self) -> dict:
        return asdict(self)


def has_token() -> bool:
    return bool(os.environ.get(TOKEN_VAR, "").strip())


def make_client(timeout: float = TIMEOUT) -> httpx.Client:
    if not has_token():
        raise ThingiverseError(f"No Thingiverse token: set {TOKEN_VAR} in .env (from thingiverse.com/apps/create).")
    # Downloads redirect to cdn.thingiverse.com; httpx drops Authorization on a redirect to another host.
    return httpx.Client(base_url=API, timeout=timeout, follow_redirects=True,
                        headers={"Authorization": f"Bearer {os.environ[TOKEN_VAR].strip()}",
                                 "User-Agent": USER_AGENT})


def _client(client: httpx.Client | None):
    """The caller's client as is, or a new one that is closed afterwards."""
    return nullcontext(client) if client is not None else make_client()


def _get(client: httpx.Client, url: str, **kw) -> httpx.Response:
    try:
        resp = client.get(url, **kw)
    except httpx.TimeoutException as e:
        raise ThingiverseError("Thingiverse did not answer in time.") from e
    except httpx.HTTPError as e:
        raise ThingiverseError(f"Could not reach Thingiverse: {e}") from e
    if resp.status_code == 401:
        raise ThingiverseError(f"Thingiverse rejected the token ({TOKEN_VAR} in .env).")
    if resp.status_code == 403:
        if "text/html" in resp.headers.get("content-type", ""):  # Cloudflare's challenge page, not the API
            raise ThingiverseError("Thingiverse's Cloudflare check refused the request (403).")
        raise ThingiverseError("Thingiverse refused the request (403); check the token.")
    if resp.status_code == 429:
        raise ThingiverseError("Thingiverse rate limit reached; try again in a minute.")
    if resp.status_code >= 400:
        raise ThingiverseError(f"Thingiverse request failed: HTTP {resp.status_code} for {resp.request.url}")
    return resp


def _json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError as e:
        raise ThingiverseError(f"Thingiverse sent a reply that is not JSON ({resp.request.url}).") from e


def _candidate(hit: dict) -> Candidate:
    return Candidate(
        id=int(hit["id"]),
        name=str(hit.get("name") or f"thing {hit['id']}"),
        url=str(hit.get("public_url") or f"https://www.thingiverse.com/thing:{hit['id']}"),
        thumbnail=str(hit.get("thumbnail") or ""),
        author=str((hit.get("creator") or {}).get("name") or "unknown"),
        license=str(hit.get("license") or "unknown license"),
        allows_derivatives=bool(hit.get("allows_derivatives")),
        is_ai=bool(hit.get("is_ai")),
        tags=[str(t.get("name") if isinstance(t, dict) else t) for t in hit.get("tags") or []],
    )


# ---------------------------------------------------------------------------------------------------------
# Is a result the requested object? (#39)
# ---------------------------------------------------------------------------------------------------------
# Thingiverse ranks on every word of the search, so "low-poly tree" returns low-poly Pokémon. A result is kept
# when its name or tags hold the object word: "tree", not "low-poly". Words that say what kind of model it is,
# or a category, are not object words.
GENERIC = {"statue", "figurine", "figure", "model", "sculpture", "toy", "print", "printable", "miniature", "mini",
           "decoration", "ornament", "low", "poly", "lowpoly", "cute", "small", "big", "large", "simple", "stylized",
           "sitting", "standing", "realistic", "cartoon", "flower", "fruit", "plant", "animal", "the", "and", "with",
           "for", "of", "a", "an", "mm"}
NAME_MATCH, TAG_MATCH = 2, 1


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _forms(word: str) -> set[str]:
    """The word in singular and plural: tree/trees, fox/foxes, butterfly/butterflies, leaf stays leaf."""
    out = {word, word + "s", word + "es"}
    if word.endswith("ies"):
        out.add(word[:-3] + "y")
    elif word.endswith("es"):
        out |= {word[:-2], word[:-1]}
    elif word.endswith("s"):
        out.add(word[:-1])
    if word.endswith("y"):
        out.add(word[:-1] + "ies")
    return out


def object_words(id_or_name: str, phrases: list[str] | str = ()) -> set[str]:
    """What a piece or request is: the words of its id or file name ("tree", "horse-statue" -> horse) that
    aren't sizes or kinds of model. If that leaves nothing, the last such word of each search phrase."""
    phrases = [phrases] if isinstance(phrases, str) else list(phrases)
    words = {w for w in _words(id_or_name) if w not in GENERIC and len(w) > 2}
    if not words:
        for phrase in phrases:
            content = [w for w in _words(phrase) if w not in GENERIC] or _words(phrase)
            words |= set(content[-1:])
    return words


def match(c: Candidate, words: set[str]) -> int:
    """NAME_MATCH when an object word (or its plural/singular) is in the name, TAG_MATCH when only in a tag."""
    forms = set().union(*(_forms(w) for w in words)) if words else set()
    if forms & set(_words(c.name)):
        return NAME_MATCH
    if forms & {w for t in c.tags for w in _words(t)}:
        return TAG_MATCH
    return 0


def search(term: str, n: int | None = None, client: httpx.Client | None = None, words: set[str] | None = None,
           strict: bool = False) -> list[Candidate]:
    """Things matching `term`, most relevant first, that have an STL small enough to download. NSFW things
    are left out. `n` None: the thingiverse_results setting.

    `words` (#39, from object_words): results whose name holds one of them come first, then tag matches, then
    the rest. With `strict`, only matches are kept, from up to POOL hits.

    Search hits have no license (the docs' example does; the real API doesn't) and no file list, so each kept
    hit costs two more requests (~0.15 s each): its files, then its details for the license and tags. With
    `strict`, details come first, so a hit that doesn't match costs one.
    """
    n = settings().thingiverse_results if n is None else n
    term = " ".join(term.split())
    if not term or n <= 0:
        return []
    strict = strict and bool(words)
    out: list[Candidate] = []
    with _client(client) as c:
        data = _json(_get(c, f"/search/{quote(term, safe='')}/",
                          params={"type": "things", "sort": "relevant", "per_page": max(n * 2, POOL) if strict else n * 2}))
        hits = data.get("hits") if isinstance(data, dict) else data  # documented: {"total", "hits"}
        for hit in hits or []:
            if strict and sum(x.match == NAME_MATCH for x in out) >= n or not strict and len(out) == n:
                break
            if not isinstance(hit, dict) or "id" not in hit or hit.get("is_nsfw"):
                continue
            if strict:
                cand = _details(c, hit)
                cand.match = match(cand, words)
                if not cand.match or not downloadable(stl_files(hit["id"], c)):
                    continue
            else:
                if not downloadable(stl_files(hit["id"], c)):  # e.g. only .obj or .3mf
                    continue
                cand = _details(c, hit)
                cand.match = match(cand, words) if words else 0
            out.append(cand)
    return sorted(out, key=lambda x: -x.match)[:n]  # stable: Thingiverse's order within each kind of match


def _details(client: httpx.Client, hit: dict) -> Candidate:
    details = _json(_get(client, f"/things/{int(hit['id'])}"))
    return _candidate({**hit, **details} if isinstance(details, dict) else hit)


def stl_files(thing_id: int, client: httpx.Client | None = None) -> list[dict]:
    with _client(client) as c:
        files = _json(_get(c, f"/things/{int(thing_id)}/files"))
    if not isinstance(files, list):
        raise ThingiverseError("Thingiverse sent an unexpected file list.")
    return [f for f in files if isinstance(f, dict) and str(f.get("name", "")).lower().endswith(".stl")]


def downloadable(files: list[dict]) -> list[dict]:
    """STL files under the size cap, largest first (usually the main body)."""
    files = [f for f in files if int(f.get("size") or 0) <= MAX_BYTES]
    return sorted(files, key=lambda f: int(f.get("size") or 0), reverse=True)


def _fetch_file(client: httpx.Client, f: dict) -> bytes:
    """The file's bytes. `download_url` (counts the download for the author) and `direct_url` (doesn't) are
    both API addresses that redirect to the CDN."""
    link = f.get("download_url") or f.get("direct_url")
    if not link:
        raise ThingiverseError(f"No download link for {f.get('name')}.")
    resp = _get(client, link)
    if "text/html" in resp.headers.get("content-type", ""):
        raise ThingiverseError(f"Got a web page instead of {f.get('name')}.")
    if len(resp.content) > MAX_BYTES:
        raise ThingiverseError(f"{f.get('name')} is larger than {MAX_BYTES // 2**20} MB.")
    return resp.content


def download_stl(thing_id: int, dest: Path, client: httpx.Client | None = None) -> tuple[Path, list[str]]:
    """Save the thing's largest STL (usually the main body) to `dest`. Returns (dest, the other STL names)."""
    with _client(client) as c:
        files = downloadable(stl_files(thing_id, c))
        if not files:
            raise ThingiverseError("This thing has no STL file small enough to download (it may only have other formats).")
        data = _fetch_file(c, files[0])
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest, [str(f["name"]) for f in files[1:]]


def _thumbnail_client() -> httpx.Client:
    """Thumbnails are public: no token (it stays with the API host)."""
    return httpx.Client(timeout=10.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})


def thumbnails(urls: list[str], client: httpx.Client | None = None) -> list[bytes | None]:
    """Each thumbnail's image bytes, or None when it can't be had. The app fetches them itself, with USER_AGENT:
    Thingiverse's image server (Cloudflare) refuses some clients, and the gallery then shows blank tiles."""

    def one(c: httpx.Client, url: str) -> bytes | None:
        if not url:
            return None
        try:
            resp = c.get(url)
        except httpx.HTTPError:
            return None
        ok = resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/")
        return resp.content if ok and len(resp.content) <= 5 * 2**20 else None

    with (nullcontext(client) if client is not None else _thumbnail_client()) as c:
        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(lambda u: one(c, u), urls))
