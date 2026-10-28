"""Paths and settings.

Settings live in one file: settings.toml in the project root (or the file named by CAD_SETTINGS).
It is per user and gitignored. On first run it is created as a copy of the built-in defaults
(data/settings.defaults.toml) with every setting commented out: to change one, uncomment its line and edit
the value. Lines left commented use the built-in default, so later changes to the defaults still apply.
Settings added in later versions are appended to the file, commented out; your lines are never changed.

Order (later wins): built-in defaults < settings.toml < `--set key=value` on the eval commands (that run only).

The `.env` file in the project root holds only API keys (ANTHROPIC_API_KEY, GEMINI_API_KEY, and the optional
THINGIVERSE_TOKEN for the Thingiverse search). Other CAD_*
variables are not read (the app warns about leftover ones), except CAD_SETTINGS, which picks another
settings file. The Anthropic SDK itself still reads ANTHROPIC_BASE_URL when providers.anthropic.base_url is unset.

Settings are read when used (`settings()`), not at import, so the app can edit the file (`set_setting`,
`unset_setting`) and the next request uses the new values. `python -m stl_agent.config` (pixi task
`settings`) lists every setting, its value and where the value came from.

Adding a setting: add it with a comment to data/settings.defaults.toml and its type to TOP_FIELDS or
PROVIDER_FIELDS below; tests/test_config.py fails if the two disagree.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
EXAMPLES_PATH = DATA_DIR / "examples.json"
FONT_DIR = DATA_DIR / "fonts"  # bundled Liberation Sans (SIL OFL, see fonts/LICENSE)
OUTPUT_DIR = ROOT / "outputs"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=value lines). Real environment variables win. The app calls it again
    to pick up keys added while it runs."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


load_dotenv()

DEFAULTS_PATH = DATA_DIR / "settings.defaults.toml"

# Schema: every setting and its type. Settings in OPTIONAL may be left unset (TOML has no null).
TOP_FIELDS = {
    "provider": str, "max_attempts": int, "api_hints": bool, "top_k": int, "retriever": str, "gate": str,
    "gate_max_rounds": int, "gate_picks": bool, "thingiverse_results": int, "thingiverse_pick": str, "judge": str, "judge_provider": str,
    "judge_threshold": (int, float),
}
PROVIDER_FIELDS = {"base_url": str, "model": str, "gate_model": str, "judge_model": str, "rpm": int,
                   "timeout": (int, float)}
OPTIONAL = {"judge_provider"}
OPTIONAL_PROVIDER = {"base_url", "gate_model", "judge_model", "rpm", "timeout"}
_LOWERCASE = {"provider", "retriever", "gate", "judge", "judge_provider", "thingiverse_pick"}


@dataclass
class Settings:
    provider: str
    max_attempts: int
    api_hints: bool
    top_k: int
    retriever: str
    gate: str
    gate_max_rounds: int
    gate_picks: bool
    thingiverse_results: int
    thingiverse_pick: str
    judge: str
    judge_threshold: float
    providers: dict[str, dict]  # name -> {"model": ..., "base_url": ...}; unset keys are missing
    judge_provider: str | None = None  # None: the build provider
    path: Path | None = None  # the settings file that was read
    sources: dict[str, str] = field(default_factory=dict)  # "providers.ollama.model" -> "default" / file name / "--set"


def settings_path() -> Path:
    return Path(os.environ.get("CAD_SETTINGS") or ROOT / "settings.toml")


def ignored_env_vars() -> list[str]:
    """CAD_* variables that are set but no longer read (settings moved to settings.toml)."""
    return sorted(k for k in os.environ if k.startswith("CAD_") and k != "CAD_SETTINGS")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _looks_secret(key: str) -> bool:
    return any(word in key.lower() for word in ("key", "token", "secret", "password"))


def _check(data: dict, fields: dict, where: str, prefix: str) -> None:
    for key, value in data.items():
        name = prefix + key
        if _looks_secret(key):
            raise ValueError(f"{where}: {name!r} looks like an API key; put keys in .env, not the settings file.")
        if key not in fields:
            raise ValueError(f"{where}: unknown setting {name!r}; known: {', '.join(prefix + k for k in fields)}")
        types = fields[key]
        # bool is a subclass of int in Python; true/false is not a number here
        if not isinstance(value, types) or (isinstance(value, bool) and types is not bool):
            raise ValueError(f"{where}: {name!r} has the wrong type ({type(value).__name__}: {value!r}).")


def validate_settings_data(data: dict, where: str, providers: set[str] | None = None) -> dict:
    """Raise ValueError naming the key for unknown keys, wrong types, API keys and unknown providers.
    `providers`: the known provider names (None: any)."""
    for key, value in data.items():
        if key != "providers":
            _check({key: value}, TOP_FIELDS, where, "")
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{where}: 'providers' must be a table like [providers.ollama].")
        for name, values in value.items():
            if providers is not None and name not in providers:
                raise ValueError(f"{where}: unknown provider 'providers.{name}'; known: {', '.join(sorted(providers))}")
            if not isinstance(values, dict):
                raise ValueError(f"{where}: 'providers.{name}' must be a table like [providers.{name}].")
            _check(values, PROVIDER_FIELDS, where, f"providers.{name}.")
    if providers is not None:
        for key in ("provider", "judge_provider"):
            value = data.get(key)
            if isinstance(value, str) and value.lower() not in providers:
                raise ValueError(f"{where}: {key} = {value!r} is not a known provider; use one of {', '.join(sorted(providers))}")
    return data


def _parse_toml(text: str, where) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{where}: not valid TOML ({e})") from None


@cache
def defaults() -> dict:
    """The built-in defaults (data/settings.defaults.toml)."""
    return validate_settings_data(_parse_toml(DEFAULTS_PATH.read_text(), DEFAULTS_PATH), str(DEFAULTS_PATH))


def known_providers() -> set[str]:
    return set(defaults()["providers"])


# ---------------------------------------------------------------------------
# Dotted names: "top_k" or "providers.ollama.model"
# ---------------------------------------------------------------------------
def _split(dotted: str) -> tuple[str, str]:
    """"top_k" -> ("", "top_k"); "providers.ollama.model" -> ("providers.ollama", "model")."""
    parts = [p.strip() for p in dotted.strip().split(".")]
    if len(parts) == 1 and parts[0] and parts[0] != "providers":
        return "", parts[0]
    if len(parts) == 3 and parts[0] == "providers" and all(parts):
        return f"providers.{parts[1].lower()}", parts[2]
    raise ValueError(f"unknown setting {dotted!r}; use a name like 'top_k' or 'providers.ollama.model'")


def _get(data: dict, table: str, key: str):
    if not table:
        return data.get(key)
    return data.get("providers", {}).get(table.split(".", 1)[1], {}).get(key)


def _with(data: dict, table: str, key: str, value) -> dict:
    """A copy of `data` with one setting set (value not None) or removed (None)."""
    data = copy.deepcopy(data)
    target = data if not table else data.setdefault("providers", {}).setdefault(table.split(".", 1)[1], {})
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value
    return data


def _values(data: dict) -> dict:
    """{dotted name: value}; empty tables don't count."""
    return dict(_flat(data))


def _flat(data: dict):
    """(dotted name, value) for every setting in `data`."""
    for key, value in data.items():
        if key == "providers":
            for name, values in value.items():
                for k, v in values.items():
                    yield f"providers.{name}.{k}", v
        else:
            yield key, value


# ---------------------------------------------------------------------------
# The settings file as lines (comments are kept; only our simple format is edited)
# ---------------------------------------------------------------------------
_HEADER = re.compile(r"^\s*\[(?!\[)\s*([^\]]+?)\s*\]\s*(#.*)?$")
_SETTING = re.compile(r"^(\s*)(#\s*)?([A-Za-z0-9_-]+)\s*=\s*(.*?)\s*$")

USER_HEADER = """\
# Your settings for the STL agent. Every setting is listed below with its default, commented out.
# To change one, delete the "# " in front of it and edit the value. Lines left commented use the
# built-in default (stl_agent/data/settings.defaults.toml), so changes to the defaults still apply.
# API keys don't go here: put them in .env. Settings added in later versions are appended, commented out.
"""


def _table_of(line: str) -> str | None:
    m = _HEADER.match(line)
    return ".".join(p.strip().strip('"') for p in m.group(1).split(".")) if m else None


def _fields_for(table: str) -> dict:
    if not table:
        return TOP_FIELDS
    return PROVIDER_FIELDS if table.startswith("providers.") and table.count(".") == 1 else {}


def _setting_lines(lines: list[str]):
    """(line index, table, key, commented) for each line that sets, or shows commented out, a known setting."""
    table = ""
    for i, line in enumerate(lines):
        t = _table_of(line)
        if t is not None:
            table = t
            continue
        m = _SETTING.match(line)
        if m and m.group(3) in _fields_for(table):
            yield i, table, m.group(3), bool(m.group(2))


def _sections(lines: list[str]) -> dict[str, tuple[int, int]]:
    """table -> (start, end) line range; "" is the part before the first table header."""
    out, table, start = {}, "", 0
    for i, line in enumerate(lines):
        t = _table_of(line)
        if t is not None:
            out.setdefault(table, (start, i))
            table, start = t, i
    out.setdefault(table, (start, len(lines)))
    return out


def _insert_at(lines: list[str], table: str) -> int | None:
    """Where a new line for `table` goes: after its last setting line, else at the end of the section."""
    sections = _sections(lines)
    if table not in sections:
        return None
    start, end = sections[table]
    last = [i for i, t, _, _ in _setting_lines(lines) if t == table]
    if last:
        return last[-1] + 1
    while end > start + (1 if table else 0) and not lines[end - 1].strip():
        end -= 1
    return end


def commented_defaults() -> str:
    """The text of a new settings file: the defaults file with every setting commented out."""
    lines = DEFAULTS_PATH.read_text().splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):  # drop the defaults file's own header comment
        i += 1
    out = []
    for line, setting in _mark_settings(lines[i:]):
        out.append(f"{setting[0]}# {line[len(setting[0]):]}" if setting and not setting[1] else line)
    return USER_HEADER + "\n".join(out) + "\n"


def _mark_settings(lines: list[str]):
    """(line, (indent, commented) or None) for each line."""
    table = ""
    for line in lines:
        t = _table_of(line)
        if t is not None:
            table = t
            yield line, None
            continue
        m = _SETTING.match(line)
        yield line, ((m.group(1), bool(m.group(2))) if m and m.group(3) in _fields_for(table) else None)


def _default_blocks() -> dict[str, list[tuple[str, list[str]]]]:
    """table -> [(key, lines)]: each default setting's comment lines and its commented-out line."""
    blocks: dict[str, list] = {"": []}
    table, pending = "", []
    for line in commented_defaults().splitlines()[USER_HEADER.count("\n"):]:
        t = _table_of(line)
        if t is not None:
            table, pending = t, []
            blocks.setdefault(table, [])
            continue
        m = _SETTING.match(line)
        if m and m.group(3) in _fields_for(table):
            blocks[table].append((m.group(3), pending + [line]))
            pending = []
        elif line.strip().startswith("#"):
            pending.append(line)
        else:
            pending = []
    return blocks


def _add_missing(text: str) -> str | None:
    """`text` with every default setting it doesn't mention added as a commented line; None if none missing."""
    lines = text.splitlines()
    data = tomllib.loads(text)
    present = {(t, k) for _, t, k, _ in _setting_lines(lines)}
    added = False
    for table, entries in _default_blocks().items():
        missing = [block for key, block in entries if (table, key) not in present and _get(data, table, key) is None]
        if not missing:
            continue
        at = _insert_at(lines, table)
        new = [line for block in missing for line in ["", *block]]
        if at is None:
            if table and table.split(".", 1)[1] in data.get("providers", {}):
                continue  # the table is written another way (e.g. dotted keys); don't declare it twice
            lines += ["", f"[{table}]", *new[1:]]
        else:
            lines[at:at] = new
        added = True
    if not added:
        return None
    new_text = "\n".join(lines) + "\n"
    try:
        same = _values(tomllib.loads(new_text)) == _values(data)  # comments only: values must not change
    except tomllib.TOMLDecodeError:
        same = False
    return new_text if same else None


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a valid TOML basic string
    raise TypeError(f"can't write {value!r} to the settings file")


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def _edit(dotted: str, value, path: Path | None) -> Path:
    """Set (value not None) or comment out (None) one setting in the settings file."""
    table, key = _split(dotted)
    if value is not None:
        validate_settings_data(_with({}, table, key, value), f"setting {dotted}", known_providers())
    path = Path(path or settings_path())
    text = path.read_text() if path.exists() else commented_defaults()
    old = _parse_toml(text, path)
    lines = text.splitlines()
    matches = [(i, commented) for i, t, k, commented in _setting_lines(lines) if (t, k) == (table, key)]
    active = [i for i, commented in matches if not commented]
    if value is None:
        # Put back the default's commented-out line, so the file keeps showing the default.
        shown = {(t, k): block[-1].strip() for t, entries in _default_blocks().items() for k, block in entries}
        for i in active:
            m = _SETTING.match(lines[i])
            lines[i] = m.group(1) + shown.get((table, key), f"# {lines[i][len(m.group(1)):]}")
    else:
        new_line = f"{key} = {_toml_value(value)}"
        target = active[-1] if active else (matches[0][0] if matches else None)
        if target is not None:
            lines[target] = _SETTING.match(lines[target]).group(1) + new_line
        else:
            at = _insert_at(lines, table)
            if at is None:
                lines += ["", f"[{table}]", new_line]
            else:
                lines.insert(at, new_line)
    new_text = "\n".join(lines) + "\n"
    try:
        ok = _values(tomllib.loads(new_text)) == _values(_with(old, table, key, value))
    except tomllib.TOMLDecodeError:
        ok = False
    if not ok:
        raise ValueError(f"Couldn't edit {dotted!r} in {path}: it is written in a form this editor doesn't handle "
                         "(e.g. a dotted key or an inline table). Edit the file by hand.")
    if new_text != text:
        _write(path, new_text)
    reload_settings()
    return path


def set_setting(dotted: str, value, path: Path | None = None) -> Path:
    """Set one setting in the settings file, e.g. set_setting("providers.ollama.model", "qwen3.5:9b").
    Uncomments the setting's line if it is commented out; other lines and comments are kept."""
    return _edit(dotted, value, path)


def unset_setting(dotted: str, path: Path | None = None) -> Path:
    """Comment a setting out again, so it uses the built-in default."""
    return _edit(dotted, None, path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_overrides: dict = {}  # from --set; this process only, never written


def load_settings(path: Path | None = None, create: bool = True) -> Settings:
    """Built-in defaults, then the settings file, then --set overrides. With `create`, a missing settings
    file is created (commented-out defaults) and an existing one gains any settings it doesn't list."""
    path = Path(path or settings_path())
    if create and not path.exists():
        try:
            _write(path, commented_defaults())
        except OSError:
            pass  # read-only location: run on defaults
    user = {}
    if path.exists():
        text = path.read_text()
        user = validate_settings_data(_parse_toml(text, path), str(path), known_providers())
        if create:
            new_text = _add_missing(text)
            if new_text:
                try:
                    _write(path, new_text)
                except OSError:
                    pass
    merged, sources = copy.deepcopy(defaults()), {}
    for dotted, _ in _flat(merged):
        sources[dotted] = "default"
    for layer, label in ((user, path.name), (_overrides, "--set")):
        for dotted, value in _flat(layer):
            merged = _with(merged, *_split(dotted), value)
            sources[dotted] = label
    top = {k: (v.lower() if k in _LOWERCASE else v) for k, v in merged.items() if k != "providers"}
    top["judge_threshold"] = float(top["judge_threshold"])
    return Settings(**top, providers=merged["providers"], path=path if path.exists() else None, sources=sources)


def _parse_value(raw: str):
    """A --set value: a TOML value when it is one (3, true, "x"), else the text itself."""
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw
    return raw if isinstance(value, (dict, list)) else value


def override_settings(pairs: list[str]) -> Settings:
    """Apply `key=value` overrides (from --set) for this process. They are not written to the file."""
    global _overrides
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set {pair!r}: expected key=value, e.g. --set providers.ollama.model=qwen3.5:9b")
        dotted, raw = (s.strip() for s in pair.split("=", 1))
        table, key = _split(dotted)
        value = _parse_value(raw)
        validate_settings_data(_with({}, table, key, value), f"--set {dotted}", known_providers())
        _overrides = _with(_overrides, table, key, value)
    return reload_settings()


def clear_overrides() -> None:
    global _overrides
    _overrides = {}


def add_set_argument(parser) -> None:
    """Add the --set option to an argparse parser; call `override_settings(args.set)` after parsing."""
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="change a setting for this run only (repeatable), e.g. --set provider=ollama "
                             "--set providers.ollama.model=qwen3.5:9b")


_settings: Settings | None = None


def settings() -> Settings:
    """Current settings (read once, until `reload_settings()`)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings() -> Settings:
    """Read the settings file again, e.g. after it changed."""
    global _settings
    _settings = None
    return settings()


def main() -> None:
    """Print every setting, its value and where the value came from."""
    s = settings()
    print(f"Settings file: {s.path or settings_path()}\n")
    rows = [(k, getattr(s, k)) for k in TOP_FIELDS]
    rows += [(f"providers.{name}.{k}", s.providers[name].get(k)) for name in s.providers for k in PROVIDER_FIELDS]
    for dotted, value in rows:
        shown = "(unset)" if value is None else _toml_value(value)
        print(f"{dotted:32} {shown:58} {s.sources.get(dotted, '')}")
    if ignored_env_vars():
        print(f"\nIgnored environment variables (settings now live in {settings_path().name}): "
              + ", ".join(ignored_env_vars()))


if __name__ == "__main__":
    main()
