#!/usr/bin/env python3
"""
Generate Sphinx-ready .rst files

Used instead of Sphinx to avoid importing broken or expensive dependecies

Usage:
    python docs/generate_rst.py qstack/ -o docs/source/ --project "Qstack" --package-root-name qstack

Execute it before making html
"""
import argparse
import ast
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable

# --------------------------- Utilities ---------------------------------

def rel_module_name(py_path: Path, package_root: Path, package_root_name: str | None) -> str:
    rel = py_path.with_suffix("").relative_to(package_root)
    parts = list(rel.parts)
    if package_root_name:
        return ".".join([package_root_name, *parts])
    return ".".join(parts)


def is_package_dir(p: Path) -> bool:
    return p.is_dir() and (p / "__init__.py").exists()


def iter_python_files(root: Path, exclude: list[str]) -> Iterable[Path]:
    exclude_patterns = [re.compile(fnmatch_to_regex(x)) for x in exclude]
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # filter dirs in-place
        dirnames[:] = [d for d in dirnames if not any(r.search(str(dp / d)) for r in exclude_patterns)]
        for fn in filenames:
            p = dp / fn
            if p.suffix == ".py" and not any(r.search(str(p)) for r in exclude_patterns):
                yield p


def fnmatch_to_regex(pattern: str) -> str:
    # crude glob->regex for our exclude filters
    escaped = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return f"^{escaped}$"

# --------------------------- AST extraction -----------------------------

@dataclass
class FunctionInfo:
    name: str
    lineno: int
    signature: str
    doc: str | None = None

@dataclass
class ClassInfo:
    name: str
    lineno: int
    doc: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)

@dataclass
class ModuleInfo:
    name: str
    path: Path
    doc: str | None
    classes: list[ClassInfo]
    functions: list[FunctionInfo]


def format_signature(args: ast.arguments) -> str:
    """Reconstruct a best-effort Python signature from ast.arguments."""
    def fmt_arg(a: ast.arg, default: str | None=None, annotation: str | None=None):
        name = a.arg
        ann = f": {annotation}" if annotation else ""
        if default is not None:
            return f"{name}{ann}={default}"
        return f"{name}{ann}"

    def ann_to_str(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        try:
            return ast.unparse(node)  # Python 3.9+
        except Exception:
            return None

    def const_to_str(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    parts: list[str] = []

    posonly = getattr(args, 'posonlyargs', []) or []
    arg_defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
    posonly_defaults = [None] * (len(posonly))

    # pos-only
    for a, d in zip(posonly, posonly_defaults, strict=True):
        parts.append(fmt_arg(a, default=const_to_str(d), annotation=ann_to_str(a.annotation)))
    if posonly:
        parts.append("/")

    # positional-or-keyword
    for a, d in zip(args.args, arg_defaults, strict=True):
        parts.append(fmt_arg(a, default=const_to_str(d), annotation=ann_to_str(a.annotation)))

    # vararg / kw-only separator
    if args.vararg:
        ann = ann_to_str(args.vararg.annotation)
        parts.append("*" + (args.vararg.arg if not ann else f"{args.vararg.arg}: {ann}"))
    elif args.kwonlyargs:
        parts.append("*")

    # kw-only
    for a, d in zip(args.kwonlyargs, args.kw_defaults or [None]*len(args.kwonlyargs), strict=True):
        parts.append(fmt_arg(a, default=const_to_str(d), annotation=ann_to_str(a.annotation)))

    # **kwargs
    if args.kwarg:
        ann = ann_to_str(args.kwarg.annotation)
        parts.append("**" + (args.kwarg.arg if not ann else f"{args.kwarg.arg}: {ann}"))

    return "(" + ", ".join([p for p in parts if p]) + ")"


def extract_module_info(py_path: Path, module_name: str) -> ModuleInfo:
    src = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ModuleInfo(module_name, py_path, None, [], [])

    mdoc = ast.get_docstring(tree)
    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cdoc = ast.get_docstring(node)
            methods: list[FunctionInfo] = []
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = safe_sig(n)
                    methods.append(FunctionInfo(n.name, n.lineno, sig, ast.get_docstring(n)))
            classes.append(ClassInfo(node.name, node.lineno, cdoc, sorted(methods, key=lambda m: m.lineno)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = safe_sig(node)
            functions.append(FunctionInfo(node.name, node.lineno, sig, ast.get_docstring(node)))

    classes.sort(key=lambda c: c.lineno)
    functions.sort(key=lambda f: f.lineno)

    return ModuleInfo(module_name, py_path, mdoc, classes, functions)


def safe_sig(fn: ast.AST) -> str:
    try:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return format_signature(fn.args)
    except Exception:
        pass
    return "(...)"

# --------------------------- RST rendering ------------------------------

SECTION_CHARS = ["=", "-", "~", ":", ".", "^"]

def title(text: str, level: int = 0) -> str:
    ch = SECTION_CHARS[level % len(SECTION_CHARS)]
    line = ch * len(text)
    return f"{text}\n{line}\n\n"


def rst_escape_heading(text: str | None) -> str:
    if not text:
        return ""
    # Minimal escaping for common problem characters in headings
    return text.replace("*", "\\*").replace("_", "\\_")


def format_docstring(doc: str | None) -> str:
    """Render docstrings as literal blocks so Sphinx won't parse them.

    This avoids common docutils errors caused by Markdown or mixed formatting
    inside Python docstrings. We also dedent to normalize indentation.
    """
    if not doc:
        return ""
    d = textwrap.dedent(doc).strip("\n")
    return "::\n\n" + textwrap.indent(d + "\n", "    ") + "\n"


def render_module_rst(mi: ModuleInfo) -> str:
    out: list[str] = []
    out.append(title(rst_escape_heading(mi.name), 0))
    out.append(format_docstring(mi.doc))

    if mi.functions:
        out.append(title("Functions", 1))
        for f in mi.functions:
            out.append(title(rst_escape_heading(f"{f.name} {f.signature}"), 2))
            out.append(format_docstring(f.doc) if f.doc else "(No docstring.)\n\n")

    if mi.classes:
        out.append(title("Classes", 1))
        for c in mi.classes:
            out.append(title(rst_escape_heading(c.name), 2))
            out.append(format_docstring(c.doc) if c.doc else "(No docstring.)\n\n")
            if c.methods:
                out.append(title("Methods", 3))
                for m in c.methods:
                    out.append(title(rst_escape_heading(f"{m.name} {m.signature}"), 4))
                    out.append(format_docstring(m.doc) if m.doc else "(No docstring.)\n\n")

    # Footer hint
    out.append(".. note::\n   Generated statically from source by gen_rst.py; no imports performed.\n")
    return "".join(out)



def render_index_rst(project: str, modules: list[ModuleInfo], out_dir: Path) -> str:
    out: list[str] = []
    heading = f"Welcome to {project} Documentation"
    out.append(title(heading, 0))

    out.append(".. include:: ../../README.md\n   :parser: myst_parser.sphinx_\n\n")

    #out.append("To do List\n==========\n\n")
    #out.append(".. todolist::\n\n")

    out.append(".. toctree::\n   :maxdepth: 1\n   :caption: Modules\n\n")

    for mi in sorted(modules, key=lambda m: m.name):
        rel = (out_path_for_module(mi, out_dir).relative_to(out_dir)).with_suffix("")
        out.append(f"   {rel.as_posix()}\n")

    out.append("\n")
    return "".join(out)



def out_path_for_module(mi: ModuleInfo, out_dir: Path) -> Path:
    # map module dotted name to file path
    parts = mi.name.split('.')
    return out_dir.joinpath(*parts).with_suffix(".rst")

# --------------------------- Main CLI -----------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("package_root", type=Path, help="Path to your package root (dir containing your modules)")
    p.add_argument("-o", "--out", dest="out_dir", type=Path, required=True, help="Output directory for .rst files")
    p.add_argument("--project", default="Project", help="Project name for index title")
    p.add_argument("--package-root-name", default=None, help="If your import package name differs from folder, set it")
    p.add_argument("--exclude", nargs="*", default=["tests", "build", "dist", "venv", "_build"],
                   help="Glob-ish patterns to exclude (matched against paths)")
    args = p.parse_args(argv)

    pkg_root = args.package_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pkg_root.exists() or not pkg_root.is_dir():
        print(f"ERROR: {pkg_root} is not a directory", file=sys.stderr)
        return 2

    # Collect modules
    modules: list[ModuleInfo] = []
    for py in iter_python_files(pkg_root, args.exclude):
        # Skip __init__.py as a module page? We'll still document it under package name
        mod_name = rel_module_name(py, pkg_root, args.package_root_name)
        mi = extract_module_info(py, mod_name)
        modules.append(mi)

    # Write each module page
    for mi in modules:
        out_path = out_path_for_module(mi, out_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_module_rst(mi), encoding="utf-8")

    # Write package-level index
    index_path = out_dir / "index.rst"
    index_path.write_text(render_index_rst(args.project, modules, out_dir), encoding="utf-8")

    print(f"Wrote {len(modules)} module pages under {out_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

