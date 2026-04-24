"""Generate Markdown API reference pages for the Zensical docs build."""

from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
from dataclasses import dataclass
from inspect import Signature
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_DIR = ROOT / "docs" / "api"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dasjax  # noqa: E402


@dataclass(frozen=True)
class ApiEntry:
    """One generated API documentation page."""

    name: str
    display_name: str
    obj: Any
    kind: str
    path: Path
    owner: "ApiEntry | None" = None


def _doc(obj: Any) -> str:
    """Return a cleaned docstring or fallback text."""
    return inspect.getdoc(obj) or "No public docstring is available."


def _summary(obj: Any) -> str:
    """Return a compact one-line summary from a docstring."""
    lines = [line.strip() for line in _doc(obj).splitlines() if line.strip()]
    return lines[0] if lines else ""


def _signature(obj: Any) -> str:
    """Return a readable signature for functions, methods, and classes."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return "()"
    params = [
        parameter
        for name, parameter in signature.parameters.items()
        if name not in {"self", "cls"}
    ]
    return str(signature.replace(parameters=params, return_annotation=Signature.empty))


def _slug(value: str) -> str:
    """Return a filesystem-safe, deterministic slug."""
    value = value.replace("__", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-").replace(".", "-").lower()


def _table_cell(value: str) -> str:
    """Escape Markdown table cell separators."""
    return value.replace("|", "\\|").replace("\n", " ")


def _page_link(from_page: Path, entry: ApiEntry) -> str:
    """Return a relative Markdown link from one generated page to another."""
    return Path(os.path.relpath(entry.path, from_page.parent)).as_posix()


def _xref_map(entries: tuple[ApiEntry, ...]) -> dict[str, ApiEntry]:
    """Return local reference aliases mapped to generated API pages."""
    refs: dict[str, ApiEntry] = {}
    for entry in entries:
        aliases = {
            entry.name,
            entry.display_name,
            entry.name.removeprefix("dasjax."),
            entry.display_name.rsplit(".", 1)[-1],
        }
        module_name = getattr(entry.obj, "__module__", None)
        object_name = getattr(entry.obj, "__name__", None)
        if module_name and object_name:
            aliases.add(f"{module_name}.{object_name}")
            aliases.add(object_name)
        if entry.kind == "method" and entry.owner is not None:
            method_name = entry.display_name.rsplit(".", 1)[-1]
            aliases.add(f"{entry.owner.display_name}.{method_name}")
            aliases.add(f"{entry.owner.name}.{method_name}")
        for alias in aliases:
            if alias and alias not in refs:
                refs[alias] = entry
    return refs


def _link_ref(token: str, *, page: Path, refs: dict[str, ApiEntry]) -> str | None:
    """Return a Markdown link for a reference token if it can be resolved."""
    clean = token.strip()
    suffix = ""
    while clean.endswith("()"):
        clean = clean[:-2]
        suffix += "()"
    entry = refs.get(clean)
    if entry is None:
        return None
    return f"[`{token}`]({_page_link(page, entry)})"


def _link_docstring(text: str, *, page: Path, refs: dict[str, ApiEntry]) -> str:
    """Convert common Python docstring references into Markdown links."""

    def _replace_role(match: re.Match[str]) -> str:
        label = match.group("label") or match.group("target")
        target = match.group("target")
        link = _link_ref(target, page=page, refs=refs)
        if link is None:
            return match.group(0)
        href = link.rsplit("](", 1)[1][:-1]
        return f"[`{label}`]({href})"

    def _replace_double_backtick(match: re.Match[str]) -> str:
        token = match.group("token")
        return _link_ref(token, page=page, refs=refs) or f"`{token}`"

    def _replace_backtick(match: re.Match[str]) -> str:
        token = match.group("token")
        return _link_ref(token, page=page, refs=refs) or match.group(0)

    text = re.sub(
        r":(?:class|func|meth|mod|obj|py:class|py:func|py:meth|py:mod):"
        r"`(?:(?P<label>[^`<>]+)\s*<)?(?P<target>[^`<>]+)>?`",
        _replace_role,
        text,
    )
    text = re.sub(r"``(?P<token>[^`]+)``", _replace_double_backtick, text)
    return re.sub(
        r"(?<!\[)`(?P<token>[A-Za-z_][A-Za-z0-9_.]*(?:\(\))?)`",
        _replace_backtick,
        text,
    )


def _kind(obj: Any) -> str:
    """Classify a documented Python object."""
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.ismethod(obj):
        return "method"
    if inspect.isfunction(obj):
        return "function"
    return "object"


def _is_owned_method(cls: type, name: str, obj: Any) -> bool:
    """Return True for public methods defined directly on a class."""
    if name.startswith("_"):
        return False
    raw = cls.__dict__.get(name)
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    return inspect.isfunction(raw)


def _owned_methods(entry: ApiEntry, method_entries: dict[tuple[type, str], ApiEntry]):
    """Yield generated pages for public methods owned by a class entry."""
    if not inspect.isclass(entry.obj):
        return ()
    rows = []
    for name, obj in sorted(entry.obj.__dict__.items()):
        if _is_owned_method(entry.obj, name, obj):
            rows.append(method_entries[(entry.obj, name)])
    return tuple(rows)


def _module_members(
    module: ModuleType,
    entries_by_obj: dict[int, ApiEntry],
    entries: tuple[ApiEntry, ...],
):
    """Yield documented public members owned by a module."""
    if module is dasjax:
        members = []
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            entry = entries_by_obj.get(id(obj))
            if entry is not None:
                members.append(entry)
        return tuple(members)

    members: list[ApiEntry] = []
    module_name = module.__name__
    for entry in entries:
        if entry.kind == "module" or entry.owner is not None:
            continue
        if getattr(entry.obj, "__module__", None) == module_name:
            members.append(entry)
    return tuple(members)


def _index_rows(entries: tuple[ApiEntry, ...], index_page: Path) -> list[str]:
    """Render the single compact API index table."""
    rows = ["| Name | Description |", "|---|---|"]
    for entry in _index_entries(entries):
        href = _page_link(index_page, entry)
        rows.append(
            f"| [`{entry.display_name}`]({href}) | {_table_cell(_summary(entry.obj))} |"
        )
    return rows


def _index_entries(entries: tuple[ApiEntry, ...]) -> tuple[ApiEntry, ...]:
    """Return entries shown in the top-level API index table."""
    return tuple(entry for entry in entries if entry.kind != "method")


def _entry_counts(entries: tuple[ApiEntry, ...]) -> dict[str, int]:
    """Return generated entry counts by kind."""
    kinds = {"module": 0, "class": 0, "function": 0, "method": 0, "object": 0}
    for entry in entries:
        kinds[entry.kind] = kinds.get(entry.kind, 0) + 1
    return kinds


def _owned_label(entry: ApiEntry) -> str:
    """Return a short label for owned-object tables."""
    if entry.kind == "method" and entry.owner is not None:
        return entry.display_name.rsplit(".", 1)[-1]
    return entry.display_name


def _owned_table(
    *,
    title: str,
    entries: tuple[ApiEntry, ...],
    page: Path,
    empty_text: str,
) -> list[str]:
    """Render a linked table for objects owned by this API object."""
    lines = [f"## {title}", ""]
    if not entries:
        lines.extend([empty_text, ""])
        return lines
    lines.extend(["| Name | Description |", "|---|---|"])
    for entry in entries:
        href = _page_link(page, entry)
        lines.append(
            f"| [`{_owned_label(entry)}`]({href}) | "
            f"{_table_cell(_summary(entry.obj))} |"
        )
    lines.append("")
    return lines


def _render_page(
    entry: ApiEntry,
    *,
    entries: tuple[ApiEntry, ...],
    entries_by_obj: dict[int, ApiEntry],
    method_entries: dict[tuple[type, str], ApiEntry],
    refs: dict[str, ApiEntry],
) -> str:
    """Render one formatted API detail page."""
    owner_link = (
        f"[`{entry.owner.display_name}`]({_page_link(entry.path, entry.owner)})"
        if entry.owner is not None
        else "None"
    )
    lines = [
        f"# {entry.display_name}",
        "",
        "<!-- This file is generated by scripts/build_api_docs.py. -->",
        "",
        '<div class="grid cards" markdown>',
        "",
        f"- __Kind__  \n  `{entry.kind}`",
        f"- __Owner__  \n  {owner_link}",
        "",
        "</div>",
        "",
    ]
    if callable(entry.obj) and not inspect.ismodule(entry.obj):
        lines.extend(
            [
                "## Signature",
                "",
                f"```python\n{entry.display_name}{_signature(entry.obj)}\n```",
                "",
            ]
        )
    doc = _link_docstring(dedent(_doc(entry.obj)), page=entry.path, refs=refs)
    lines.extend(['!!! abstract "Description"', ""])
    lines.extend(f"    {line}" if line else "" for line in doc.splitlines())
    lines.append("")

    if inspect.ismodule(entry.obj):
        lines.extend(
            _owned_table(
                title="Owned Public API",
                entries=_module_members(entry.obj, entries_by_obj, entries),
                page=entry.path,
                empty_text="This module has no documented owned public API entries.",
            )
        )
    elif inspect.isclass(entry.obj):
        owned_methods = _owned_methods(entry, method_entries)
        if owned_methods:
            lines.extend(
                _owned_table(
                    title="Owned Methods",
                    entries=owned_methods,
                    page=entry.path,
                    empty_text="",
                )
            )

    return "\n".join(lines)


def _collect_entries() -> tuple[
    tuple[ApiEntry, ...],
    dict[int, ApiEntry],
    dict[tuple[type, str], ApiEntry],
]:
    """Collect root, public symbol, operation, and method API pages."""
    root_module = ApiEntry(
        name="dasjax",
        display_name="dasjax",
        obj=dasjax,
        kind="module",
        path=OUT_DIR / "dasjax.md",
    )
    module_objects: dict[str, ModuleType] = {"dasjax": dasjax}
    entries: list[ApiEntry] = [root_module]

    for name in dasjax.__all__:
        obj = getattr(dasjax, name)
        module_name = getattr(obj, "__module__", None)
        if module_name:
            module = sys.modules.get(module_name)
            if isinstance(module, ModuleType):
                module_objects[module_name] = module
        entries.append(
            ApiEntry(
                name=f"dasjax.{name}",
                display_name=name,
                obj=obj,
                kind=_kind(obj),
                path=OUT_DIR / f"{_slug(name)}.md",
            )
        )

    seen_operation_classes: set[type] = set()
    for operation_name in dasjax.list_patch_operations():
        cls = dasjax.get_patch_operation(operation_name)
        module = sys.modules.get(cls.__module__)
        if isinstance(module, ModuleType):
            module_objects[cls.__module__] = module
        if cls in seen_operation_classes:
            continue
        seen_operation_classes.add(cls)
        entries.append(
            ApiEntry(
                name=f"{cls.__module__}.{cls.__name__}",
                display_name=operation_name,
                obj=cls,
                kind="class",
                path=OUT_DIR / "operations" / f"{_slug(operation_name)}.md",
            )
        )

    module_entries = [
        ApiEntry(
            name=module_name,
            display_name=module_name,
            obj=module,
            kind="module",
            path=OUT_DIR / "modules" / f"{_slug(module_name)}.md",
        )
        for module_name, module in sorted(module_objects.items())
        if module_name != "dasjax"
    ]
    entries[1:1] = module_entries

    method_entries: dict[tuple[type, str], ApiEntry] = {}
    for owner in tuple(entries):
        if not inspect.isclass(owner.obj):
            continue
        for method_name, method_obj in sorted(owner.obj.__dict__.items()):
            if not _is_owned_method(owner.obj, method_name, method_obj):
                continue
            entry = ApiEntry(
                name=f"{owner.name}.{method_name}",
                display_name=f"{owner.display_name}.{method_name}",
                obj=method_obj,
                kind="method",
                path=OUT_DIR
                / "methods"
                / f"{_slug(owner.name)}-{_slug(method_name)}.md",
                owner=owner,
            )
            method_entries[(owner.obj, method_name)] = entry
            entries.append(entry)

    entries_by_obj = {id(entry.obj): entry for entry in entries}
    return tuple(entries), entries_by_obj, method_entries


def _render_index(entries: tuple[ApiEntry, ...]) -> str:
    """Render the compact linked API index."""
    index_page = OUT_DIR / "index.md"
    counts = _entry_counts(entries)
    lines = [
        "# API Reference",
        "",
        "<!-- This file is generated by scripts/build_api_docs.py. -->",
        "",
        "This page is generated from the public `dasjax` Python API and the "
        "registered pipeline operation surface at build time.",
        "",
        '<div class="grid cards" markdown>',
        "",
        f"- __Modules__  \n  {counts['module']} generated module pages",
        f"- __Classes__  \n  {counts['class']} generated class pages",
        f"- __Functions__  \n  {counts['function']} public helper functions",
        f"- __Methods__  \n  {counts['method']} method pages linked from owners",
        "",
        "</div>",
        "",
        '!!! tip "How to use this reference"',
        "",
        "    Use the table below for public entry points, modules, and pipeline "
        "operations. Method pages are linked from each owning class page so the "
        "top-level index stays scannable.",
        "",
        *_index_rows(entries, index_page),
        "",
    ]
    return "\n".join(lines)


def _clean_output() -> None:
    """Remove old generated API pages before writing a fresh set."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Write generated API docs to the docs tree."""
    entries, entries_by_obj, method_entries = _collect_entries()
    refs = _xref_map(entries)
    _clean_output()
    (OUT_DIR / "index.md").write_text(_render_index(entries), encoding="utf-8")
    for entry in entries:
        entry.path.parent.mkdir(parents=True, exist_ok=True)
        entry.path.write_text(
            _render_page(
                entry,
                entries=entries,
                entries_by_obj=entries_by_obj,
                method_entries=method_entries,
                refs=refs,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(entries) + 1} API documentation pages")


if __name__ == "__main__":
    main()
