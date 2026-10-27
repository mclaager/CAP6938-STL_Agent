"""Thingiverse search and download (#9), for requests code generation can't build.

API: https://www.thingiverse.com/developers/swagger. Needs an app token (THINGIVERSE_TOKEN in .env, from
thingiverse.com/apps/create); without one, every request is refused. Generation works without it.

    search(term) -> [Candidate]                  only things with a downloadable STL, with their license
    download_stl(thing_id, dest) -> (path, other STL names)

Cloudflare in front of the API and the file CDN answers 403 to default client User-Agents (curl's,
httpx's), token or not, so every request sends USER_AGENT.

Every function takes an optional `client` (an httpx.Client) so tests can pass one with a mock transport.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from ..config import settings

API = "https://api.thingiverse.com"
TOKEN_VAR = "THINGIVERSE_TOKEN"
MAX_BYTES = 100 * 1024 * 1024  # skip files larger than this
TIMEOUT = 30
USER_AGENT = "stl-agent/0.1 (CAP 6938 course project)"  # any non-default value passes Cloudflare


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
    )


def search(term: str, n: int | None = None, client: httpx.Client | None = None) -> list[Candidate]:
    """Things matching `term`, most relevant first, that have an STL small enough to download. NSFW things
    are left out. `n` None: the thingiverse_results setting.

    Search hits have no license (the docs' example does; the real API doesn't) and no file list, so each kept
    hit costs two more requests (~0.15 s each): its files, then its details for the license.
    """
    n = settings().thingiverse_results if n is None else n
    term = " ".join(term.split())
    if not term or n <= 0:
        return []
    out: list[Candidate] = []
    with _client(client) as c:
        data = _json(_get(c, f"/search/{quote(term, safe='')}/",
                          params={"type": "things", "sort": "relevant", "per_page": n * 2}))
        hits = data.get("hits") if isinstance(data, dict) else data  # documented: {"total", "hits"}
        for hit in hits or []:
            if len(out) == n:
                break
            if not isinstance(hit, dict) or "id" not in hit or hit.get("is_nsfw"):
                continue
            if not downloadable(stl_files(hit["id"], c)):  # e.g. only .obj or .3mf
                continue
            details = _json(_get(c, f"/things/{int(hit['id'])}"))
            out.append(_candidate({**hit, **details} if isinstance(details, dict) else hit))
    return out


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
