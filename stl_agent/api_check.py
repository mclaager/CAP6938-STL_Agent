"""CadQuery API hints for the repair loop: say which calls in a script are wrong and what the real ones are.

Two parts, both used only when a script fails to run (so a wrong guess here can never block working code):

- `check_code(code)`: reads the script without running it (Python's `ast`) and checks every CadQuery call
  whose object it can work out: `cq.X` names, `cq.Workplane(...)` chains and variables assigned from them.
  It finds methods that don't exist and arguments that don't fit the method's real signature (taken from the
  installed CadQuery with `inspect`), including a tuple passed where a number goes. It reports all of them at
  once, where running the script only shows the first.
- `runtime_hint(...)`: for the error the run did hit, the signature of the CadQuery method it happened in
  and, for a missing name, the closest real ones.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
import typing
from functools import lru_cache

# Wrong guesses whose right name isn't spelled alike, so close-match search can't find it.
ALIASES = {
    "radius": "round edges with .fillet(r)",
    "round": "round edges with .fillet(r)",
    "fillet_all": "select edges first, e.g. .edges().fillet(r)",
    "cylinder_hole": "use .hole(d) on a face workplane",
    "add_hole": "use .hole(d) on a face workplane",
    "cast": "build solids with Workplane methods (box, cylinder, extrude) instead of cq.Solid",
    "copy": "Workplane methods return a new Workplane; assign it to a new variable instead of copying",
    "move": "use .translate((x, y, z))",
    "rotateAboutCenter": "use .rotate((0, 0, 0), (0, 0, 1), angle)",
    "cutThruAll2": "use .cutThruAll()",
    "subtract": "use .cut(other)",
    "difference": "use .cut(other)",
    "add": "use .union(other)",
    "fuse": "use .union(other)",
}

_NUMBER_NAMES = {"float", "int", "Real", "Union[int, float]", "Union[float, int]"}


@lru_cache(maxsize=1)
def _cq():
    import cadquery

    return cadquery


def _classes() -> dict[str, type]:
    cq = _cq()
    return {n: getattr(cq, n) for n in dir(cq) if not n.startswith("_") and isinstance(getattr(cq, n), type)}


def _public(obj) -> list[str]:
    return [n for n in dir(obj) if not n.startswith("_")]


def signature_text(owner: str, name: str, fn) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"{owner}.{name}(...)"
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    parts = []
    for p in params:
        s = p.name if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) else ("*" if p.kind == p.VAR_POSITIONAL else "**") + p.name
        if p.default is not p.empty:
            d = repr(p.default)
            s += "=" + (d if len(d) <= 20 else "...")
        parts.append(s)
    return f"{owner}.{name}({', '.join(parts)})"


def missing_name_hint(owner_name: str, owner, name: str) -> str:
    """'X has no Y' plus the nearest real names, a hand-written alternative, or another class that has Y."""
    msg = f"{owner_name} has no `{name}`."
    close = difflib.get_close_matches(name, _public(owner), n=3, cutoff=0.6)
    if close:
        msg += " Did you mean: " + ", ".join(f"`{c}`" for c in close) + "?"
    if name in ALIASES:
        msg += f" ({ALIASES[name]})"
    elsewhere = [n for n, c in _classes().items() if c is not owner and hasattr(c, name) and n in ("Workplane", "Sketch")]
    if elsewhere and owner is not _cq():
        msg += " `" + name + "` exists on " + " and ".join(f"cq.{n}" for n in elsewhere) + "."
    if close and not elsewhere:
        fn = getattr(owner, close[0], None)
        if callable(fn):
            msg += f" Signature: {signature_text(owner_name.removeprefix('a ').removeprefix('cq.'), close[0], fn)}."
    return msg


# ---------------------------------------------------------------------------
# Reading the script
# ---------------------------------------------------------------------------
class _Instance:
    """'An object of class cls', as opposed to the class itself."""

    def __init__(self, cls: type):
        self.cls = cls


def _is_number_annotation(ann) -> bool:
    if ann in (float, int):
        return True
    text = ann if isinstance(ann, str) else str(ann).replace("typing.", "")
    return text in _NUMBER_NAMES


def _resolve_return(fn, receiver_cls):
    try:
        ret = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return None
    if isinstance(ret, typing.TypeVar):
        return _Instance(receiver_cls)
    if isinstance(ret, str):
        ret = _classes().get(ret.strip("'\""))
    if isinstance(ret, type) and ret in _classes().values():
        return _Instance(ret)
    return None


class _Checker(ast.NodeVisitor):
    def __init__(self):
        self.env: dict[str, object] = {"cq": _cq()}
        self.problems: list[tuple[int, str]] = []

    # --- what an expression is: the cq module, a cq class, an instance of one, or None (unknown)
    def kind(self, node):
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.kind(node.value)
            if base is _cq() or isinstance(base, type):
                if not hasattr(base, node.attr):
                    return None  # reported where it's used
                val = getattr(base, node.attr)
                return val if isinstance(val, type) or callable(val) else None
            return None
        if isinstance(node, ast.Call):
            return self.call_result(node)
        return None

    def call_result(self, node: ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute):
            base = self.kind(f.value)
            if isinstance(base, _Instance):
                fn = getattr(base.cls, f.attr, None)
                return _resolve_return(fn, base.cls) if callable(fn) else None
            if isinstance(base, type):
                fn = getattr(base, f.attr, None)
                return _resolve_return(fn, base) if callable(fn) else None
            if base is _cq():
                target = getattr(base, f.attr, None)
                return _Instance(target) if isinstance(target, type) else None
        return None

    # --- checks
    def visit_Attribute(self, node: ast.Attribute):
        base = self.kind(node.value)
        owner, owner_name = None, ""
        if base is _cq():
            owner, owner_name = base, "cq"
        elif isinstance(base, type):
            owner, owner_name = base, f"cq.{base.__name__}"
        elif isinstance(base, _Instance):
            owner, owner_name = base.cls, f"a {base.cls.__name__}"
        if owner is not None and not hasattr(owner, node.attr):
            self.problems.append((node.lineno, missing_name_hint(owner_name, owner, node.attr)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.check_args(node)
        self.generic_visit(node)

    def check_args(self, node: ast.Call):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            return  # *args / **kwargs: can't tell what is passed
        f = node.func
        if not isinstance(f, ast.Attribute):
            return
        base = self.kind(f.value)
        if isinstance(base, _Instance):
            cls, bound_self = base.cls, True
        elif isinstance(base, type):
            cls, bound_self = base, False
        elif base is _cq() and isinstance(getattr(base, f.attr, None), type):  # a constructor: cq.Workplane(...)
            cls = getattr(base, f.attr)
            return self.bind(node, cls, "__init__", cls.__name__, True)
        else:
            return
        if hasattr(cls, f.attr):
            self.bind(node, cls, f.attr, f"{cls.__name__}.{f.attr}", bound_self)

    def bind(self, node, cls, attr, label, bound_self):
        static = inspect.getattr_static(cls, attr, None)
        fn = getattr(cls, attr)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return
        params = list(sig.parameters.values())
        if bound_self and not isinstance(static, (staticmethod, classmethod)) and params and params[0].name == "self":
            sig = sig.replace(parameters=params[1:])
        try:
            bound = sig.bind(*node.args, **{k.arg: k.value for k in node.keywords})
        except TypeError as e:
            owner = "Workplane" if attr == "__init__" else label.split(".")[0]
            shown = signature_text(owner, "" if attr == "__init__" else attr, fn).replace(".(", "(")
            self.problems.append((node.lineno, f"`{label}(...)`: {e}. Signature: {shown}."))
            return
        for name, value in bound.arguments.items():
            p = sig.parameters[name]
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if _is_number_annotation(p.annotation) and isinstance(value, (ast.Tuple, ast.List, ast.Constant)) and not (
                isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool)
            ):
                got = "a tuple" if isinstance(value, ast.Tuple) else "a list" if isinstance(value, ast.List) else type(value.value).__name__
                shown = signature_text(label.split(".")[0], attr, fn)
                self.problems.append((node.lineno, f"`{label}(...)`: `{name}` must be a number, got {got}. Signature: {shown}."))

    # --- variables: straight-line, last assignment wins; anything not a plain name becomes unknown
    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        k = self.kind(node.value)
        for t in node.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    self.env[n.id] = k if isinstance(t, ast.Name) else None

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.env[node.target.id] = None

    def visit_For(self, node):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.env[n.id] = None
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        saved = dict(self.env)
        for a in node.args.args + node.args.kwonlyargs:
            self.env[a.arg] = None
        for stmt in node.body:
            self.visit(stmt)
        self.env = saved
        self.env[node.name] = None

    def visit_Import(self, node):
        for a in node.names:
            if a.name == "cadquery":
                self.env[a.asname or "cadquery"] = _cq()

    def visit_ImportFrom(self, node):
        if node.module == "cadquery":
            for a in node.names:
                self.env[a.asname or a.name] = getattr(_cq(), a.name, None)


def check_code(code: str) -> list[str]:
    """Problems with the script's CadQuery calls, one line each ("line 12: ..."), in line order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    c = _Checker()
    c.visit(tree)
    seen, out = set(), []
    for line, msg in sorted(c.problems, key=lambda p: p[0]):
        if (line, msg) not in seen:
            seen.add((line, msg))
            out.append(f"line {line}: {msg}")
    return out


# ---------------------------------------------------------------------------
# The error the run hit
# ---------------------------------------------------------------------------
_NO_ATTR = re.compile(r"'(\w+)' object has no attribute '(\w+)'|type object '(\w+)' has no attribute '(\w+)'|module '(\w+)' has no attribute '(\w+)'")


def runtime_hint(error_type: str | None, error_msg: str | None, lib_call: str | None) -> str | None:
    """One line for the error itself. `lib_call` is the qualified name of the CadQuery function the error
    happened in (e.g. "Workplane.rect"), recorded by the sandbox."""
    msg = error_msg or ""
    if error_type == "AttributeError":
        m = _NO_ATTR.search(msg)
        if m:
            owner_name, name = next((m.group(i), m.group(i + 1)) for i in (1, 3, 5) if m.group(i))
            classes = _classes()
            if owner_name in ("cadquery", "cq"):
                return missing_name_hint("cq", _cq(), name)
            if owner_name in classes:
                label = f"cq.{owner_name}" if m.group(3) else f"a {owner_name}"
                return missing_name_hint(label, classes[owner_name], name)
    if lib_call and error_type in ("TypeError", "AttributeError", "ValueError"):
        owner_name, _, name = lib_call.rpartition(".")
        owner = _classes().get(owner_name)
        fn = getattr(owner, name, None) if owner else None
        if callable(fn):
            return f"The error happened inside CadQuery's {owner_name}.{name}; its signature is {signature_text(owner_name, name, fn)}."
    return None


def hints(code: str, error_type: str | None, error_msg: str | None, lib_call: str | None) -> str:
    """Extra repair feedback for a failed run: the runtime hint, then every problem found by reading the code."""
    lines = []
    h = runtime_hint(error_type, error_msg, lib_call)
    if h:
        lines.append(h)
    problems = check_code(code)
    if problems:
        lines.append("Problems found in the script's CadQuery calls (fix all of them, not only the one above):")
        lines += [f"- {p}" for p in problems]
    return "\n".join(lines)
