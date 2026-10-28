"""requirements.txt (for pip users) lists the same packages as pixi.toml's cap6938 feature."""

import re
import tomllib

from stl_agent.config import ROOT


def names(specs) -> set[str]:
    return {re.split(r"[<>=!~\[ ;]", s.strip(), maxsplit=1)[0].lower().replace("_", "-") for s in specs}


def test_requirements_match_pixi():
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text())["feature"]["cap6938"]["pypi-dependencies"]
    lines = [ln.split("#")[0] for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    pip = names(ln for ln in lines if ln.strip())
    assert pip == names(pixi), f"only in pixi.toml: {names(pixi) - pip}; only in requirements.txt: {pip - names(pixi)}"
