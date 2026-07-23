#!/usr/bin/env python3
"""
Documentation reference generator for Phaser.

Run as `python scripts/generate_docs.py [--output DIR]`. It rebuilds the whole
`docs/generated/` tree from scratch (removing it first) into three parts:

- `api/state.md` — the public `phaser.state` classes (unchanged since Phase 0).
- `plan/index.md` — every Pane-backed plan class reachable from
  `phaser.plan.ReconsPlan`.
- `hooks/*.md` — one page per hook-family registry, plus `hooks/index.md`.

Generation must be deterministic (see "Ordering and determinism" below) and a
failure here is a hard error — there are no fallbacks and nothing is guessed.

How a new plan class is picked up
----------------------------------
The plan reference is built by a breadth-first walk starting at
`phaser.plan.ReconsPlan`'s Pane fields (`cls.__pane_info__.fields`, which
already includes inherited fields — see `pane.classes._process`). Whenever a
field's type annotation contains another Pane dataclass (a subclass of
`phaser.types.Dataclass`), that class is queued and rendered as its own
section on the same page. This is how `InitPlan`, `SaveOptions`, `SimpleFlag`,
and the `Slices` union members (`SliceList`/`SliceStep`/`SliceTotal`) are
discovered without being hand-listed anywhere.

`EngineHook` is special-cased: its `known` registry maps built-in engine names
to concrete engine-plan classes (`ConventionalEnginePlan`, `GradientEnginePlan`),
and those classes are walked exactly like any other nested plan class. A new
engine registered in `EngineHook.known` with a `Dataclass` props type is picked
up automatically. Fields typed as any *other* `Hook` subclass (noise models,
solvers, constraints, schedules, loaders, ...) are deliberately **not**
expanded inline in the plan reference — they are linked to the matching page
under `hooks/`, which is the single source for hook property schemas. If you
add a field to an existing plan class, it appears the next time this script
runs; nothing needs to be registered with the generator itself.

How a new hook registry is picked up
-------------------------------------
Every hook family is a subclass of `phaser.hooks.hook.Hook` carrying a
class-level `known: dict[str, tuple]` registry. This script discovers every
registry with a recursive `Hook.__subclasses__()` walk, run after importing
`phaser.plan` — that import transitively imports every module that currently
registers a hook family (`phaser/hooks/__init__.py`, `.solver`, `.schedule`,
`.regularization`, plus the registrations `phaser/plan.py` performs itself).

**If your new hook family's page does not appear:** first check that the
module defining it is actually imported somewhere reachable from
`phaser.plan` — a `Hook` subclass Python has never imported cannot be found by
`__subclasses__()`. Then check that you added an entry to that family's
`known` dict, e.g. `MyHook.known['my_name'] = ('package.module:function',
MyPropsClass)` (optionally with a third tuple element, a dependency-name
tuple looked up in `phaser/hooks/_dependencies.py`). Nothing else needs to
change here — the built-in name, target, property schema, and any declared
optional dependencies are read straight from that tuple.

Ordering and determinism
-------------------------
Plan classes are rendered with `ReconsPlan` first, then every other discovered
class sorted alphabetically by class name; fields within a class keep Pane's
own field order (base-class fields before subclass-added fields, in
annotation order). Hook families are sorted alphabetically by class name;
built-in hooks within a family are grouped by identical target so that
aliases (multiple short names registered to the same function and props
class) share one entry, and groups are sorted alphabetically by their first
name. No output may depend on dict iteration order, object identity, or
memory addresses; a `repr()` that looks non-deterministic (contains a hex
address) is either special-cased by type (see `_describe_metadata`) or
omitted — never guessed at.

Runtime-default notes (curated, not generated)
------------------------------------------------
Some schema defaults are sentinels resolved by engine code at run time, not
by the schema itself (`shuffle_groups: None`, `grouping: None`, ...). These
cannot be introspected from `phaser/plan.py` alone, so they live in
`scripts/generated_docs_notes.py` as a hand-maintained, cited dict. This
generator looks a note up while rendering a field and includes it if present;
it never invents one. See that module's docstring for the key format and the
verification requirement for new entries.
"""

import argparse
import collections
import collections.abc
import dataclasses
import inspect
import pathlib
import re
import shutil
import sys
import typing as t

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pane.annotations as pane_annotations

import phaser.state
import phaser.plan as plan_module
from phaser.hooks.hook import Hook
from phaser.hooks._dependencies import _DEPENDENCIES
from phaser.types import Dataclass, _EmptyDictAnnotation, _ReconsVarsAnnotation

from generated_docs_notes import RUNTIME_DEFAULT_NOTES


PUBLIC_STATE_CLASSES = (
    phaser.state.Patterns,
    phaser.state.IterState,
    phaser.state.ProbeState,
    phaser.state.ObjectState,
    phaser.state.ProgressState,
    phaser.state.ReconsState,
    phaser.state.PartialReconsState,
    phaser.state.PreparedRecons,
)


# ---------------------------------------------------------------------------
# Shared helpers (used by the state, plan, and hooks generators)
# ---------------------------------------------------------------------------


def format_annotation(annotation: t.Any) -> str:
    if annotation is dataclasses.MISSING:
        return "Unknown"
    return inspect.formatannotation(annotation).replace("typing.", "")


def first_line(doc: t.Optional[str]) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ")


def field_docs(cls: type) -> dict[str, str]:
    """Extract field docstrings (trailing string literals after an annotated
    assignment) from `cls`'s *own* source — does not look at base classes."""
    source = inspect.getsource(cls)
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    docs: dict[str, str] = {}
    body = tree.body[0].body
    for index, node in enumerate(body[:-1]):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        next_node = body[index + 1]
        if (
            isinstance(next_node, ast.Expr)
            and isinstance(next_node.value, ast.Constant)
            and isinstance(next_node.value.value, str)
        ):
            docs[node.target.id] = " ".join(next_node.value.value.split())
    return docs


def field_docs_all(cls: type) -> dict[str, str]:
    """Like `field_docs`, but merged across `cls`'s MRO, so a field's docstring
    is found even when the field is only defined on a base class (e.g.
    `EnginePlan.buffer_n_groups` rendered under `ConventionalEnginePlan`).
    A more-derived class's own docstring wins over an inherited one."""
    merged: dict[str, str] = {}
    for base in reversed(cls.__mro__):
        if not is_dataclass_cls(base):
            continue
        try:
            merged.update(field_docs(base))
        except (OSError, TypeError):
            continue
    return merged


def escape_cell(text: str) -> str:
    """Make free-form text (docstrings, curated notes, value reprs) safe to
    embed in a single Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_default(value: t.Any) -> str:
    return repr(value)


def is_dataclass_cls(obj: t.Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, Dataclass)


def is_hook_cls(obj: t.Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, Hook)


def iter_type_atoms(annotation: t.Any) -> t.Iterator[type]:
    """Yield every class-like atom reachable in a (possibly nested) typing
    annotation: walks Annotated/Union/Literal/generic-alias structure only,
    never a class's own field definitions."""
    if annotation is None or annotation is type(None):
        return
    if hasattr(annotation, "__metadata__"):
        yield from iter_type_atoms(annotation.__origin__)
        return
    origin = t.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            yield annotation
        return
    if origin is t.Literal:
        return
    for arg in t.get_args(annotation):
        yield from iter_type_atoms(arg)


def find_dataclasses(annotation: t.Any) -> list[type]:
    return [a for a in iter_type_atoms(annotation) if is_dataclass_cls(a)]


def find_hooks(annotation: t.Any) -> list[type]:
    return [a for a in iter_type_atoms(annotation) if is_hook_cls(a)]


def path_suffix(annotation: t.Any) -> str:
    ann = annotation
    while hasattr(ann, "__metadata__"):
        ann = ann.__origin__
    origin = t.get_origin(ann)
    if origin in (list, tuple, collections.abc.Sequence):
        return "[]"
    if origin in (dict, collections.abc.Mapping):
        return "{}"
    return ""


_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def hook_family_name(hook_cls: type) -> str:
    name = hook_cls.__name__
    if name.endswith("Hook"):
        name = name[: -len("Hook")]
    words = _CAMEL_BOUNDARY.split(name)
    return " ".join(words)


def hook_family_slug(hook_cls: type) -> str:
    return hook_family_name(hook_cls).lower().replace(" ", "-")


def _describe_metadata(meta: t.Any) -> t.Optional[str]:
    """Render one `Annotated[...]` metadata object as deterministic text, or
    `None` if it cannot be rendered deterministically (e.g. its `repr()`
    embeds a live object's memory address) — never a guess at its meaning."""
    if isinstance(meta, pane_annotations.Condition):
        return meta.name
    text = repr(meta)
    if re.search(r"0x[0-9a-fA-F]+", text):
        return None
    return text


def describe_field_type(
    annotation: t.Any,
    *,
    link_dataclasses: t.Optional[t.Set[str]],
    hooks_prefix: str,
) -> str:
    """Render a type annotation as neutral Markdown.

    `link_dataclasses`, if given, is the set of plan-class names being
    rendered on *this same page*; a nested `Dataclass` is linked to its
    same-page anchor only if its name is in that set (pass `None` to never
    link, e.g. when rendering hook property schemas, whose nested dataclasses
    are not given their own page). `Hook` subclasses are always linked to
    `{hooks_prefix}{slug}.md`, since every hook family unconditionally gets a
    generated page.
    """
    if annotation is type(None):
        return "`None`"
    if annotation is Ellipsis:
        return "..."
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, _EmptyDictAnnotation):
                return "empty mapping `{}`"
            if isinstance(meta, _ReconsVarsAnnotation):
                inner = describe_field_type(
                    annotation.__origin__, link_dataclasses=link_dataclasses, hooks_prefix=hooks_prefix
                )
                return f"{inner} (a comma-separated string is also accepted)"
        base = describe_field_type(
            annotation.__origin__, link_dataclasses=link_dataclasses, hooks_prefix=hooks_prefix
        )
        extras = [e for e in (_describe_metadata(m) for m in annotation.__metadata__) if e]
        return base if not extras else f"{base} (constraint: {'; '.join(extras)})"

    origin = t.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            if is_dataclass_cls(annotation):
                if link_dataclasses is not None and annotation.__name__ in link_dataclasses:
                    return f"[`{annotation.__name__}`](#{annotation.__name__.lower()})"
                return f"`{annotation.__name__}`"
            if is_hook_cls(annotation):
                slug = hook_family_slug(annotation)
                return f"hook: [`{annotation.__name__}`]({hooks_prefix}{slug}.md)"
            return f"`{annotation.__name__}`"
        if annotation is t.Any:
            return "`Any`"
        return f"`{annotation!r}`"

    if origin is t.Literal:
        return " \\| ".join(f"`{arg!r}`" for arg in t.get_args(annotation))
    if origin is t.Union:
        parts = [
            describe_field_type(arg, link_dataclasses=link_dataclasses, hooks_prefix=hooks_prefix)
            for arg in t.get_args(annotation)
        ]
        return " \\| ".join(parts)

    args = t.get_args(annotation)
    name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    if args:
        inner = ", ".join(
            describe_field_type(arg, link_dataclasses=link_dataclasses, hooks_prefix=hooks_prefix)
            for arg in args
        )
        return f"`{name}`[{inner}]"
    return f"`{name}`"


# ---------------------------------------------------------------------------
# phaser.state reference (unchanged behavior; output path unchanged)
# ---------------------------------------------------------------------------


def render_class(cls: type) -> list[str]:
    lines = [f"## `{cls.__name__}`", ""]
    class_doc = first_line(inspect.getdoc(cls))
    if class_doc and not class_doc.startswith(f"{cls.__name__}("):
        lines.extend((class_doc, ""))

    docs = field_docs(cls)
    fields = dataclasses.fields(cls)
    if fields:
        lines.extend(("### Fields", "", "| Name | Type | Description |", "| --- | --- | --- |"))
        for field in fields:
            description = docs.get(field.name, "")
            lines.append(
                f"| `{field.name}` | `{format_annotation(field.type)}` | {description} |"
            )
        lines.append("")

    methods = []
    for name, member in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_") or member.__module__ != cls.__module__:
            continue
        methods.append((name, member))
    if methods:
        lines.extend(("### Methods", ""))
        for name, method in methods:
            try:
                signature = inspect.signature(method)
            except ValueError:
                signature = "(...)"
            lines.append(f"- `{name}{signature}` — {first_line(inspect.getdoc(method))}")
        lines.append("")

    return lines


def generate_state_reference(output: pathlib.Path) -> None:
    lines = [
        "# `phaser.state`",
        "",
        "<!-- Generated by scripts/generate_docs.py. Do not edit. -->",
        "",
        "This reference is generated from the public state classes in `phaser/state.py`.",
        "",
    ]
    for cls in PUBLIC_STATE_CLASSES:
        lines.extend(render_class(cls))
    output.write_text("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Plan reference (docs/generated/plan/index.md)
# ---------------------------------------------------------------------------

_ROOT_LABEL = "(top-level plan)"


def discover_plan_graph(root: type) -> tuple[list[type], dict[type, list[str]]]:
    """Breadth-first walk of every Pane dataclass reachable from `root`'s
    fields. Returns `(classes_in_discovery_order, usage)`, where `usage[cls]`
    is every YAML path at which `cls` was encountered (a class reused by
    multiple parents, like `SaveOptions` or the `Slices` union members, gets
    every path). See the module docstring for the `EngineHook` special case.
    """
    usage: dict[type, list[str]] = {}
    order: list[type] = []
    seen: set[type] = set()
    queue: collections.deque = collections.deque([(root, _ROOT_LABEL)])

    while queue:
        cls, path = queue.popleft()
        usage.setdefault(cls, [])
        if path not in usage[cls]:
            usage[cls].append(path)
        if cls in seen:
            continue
        seen.add(cls)
        order.append(cls)

        for field in cls.__pane_info__.fields:
            suffix = path_suffix(field.type)
            key = field.out_name + suffix
            child_path = key if path == _ROOT_LABEL else f"{path}.{key}"

            for nested in find_dataclasses(field.type):
                queue.append((nested, child_path))

            for hook_cls in find_hooks(field.type):
                if hook_cls is plan_module.EngineHook:
                    base = field.out_name if path == _ROOT_LABEL else f"{path}.{field.out_name}"
                    for name, entry in sorted(hook_cls.known.items()):
                        _ref, props_ty, _deps = parse_hook_entry(entry)
                        if is_dataclass_cls(props_ty):
                            queue.append((props_ty, f"{base}[type={name}]"))

    return order, usage


def ordered_plan_classes(root: type, discovered: list[type]) -> list[type]:
    rest = sorted((c for c in discovered if c is not root), key=lambda c: c.__name__)
    return [root, *rest]


def render_plan_class(
    cls: type, usage: dict[type, list[str]], class_names: t.Set[str]
) -> list[str]:
    lines = [f"## `{cls.__name__}`", ""]
    # `cls.__doc__` only (never `inspect.getdoc`, which would fall back to an
    # inherited docstring from `pane.PaneBase` and misleadingly attribute it
    # to every plan class that doesn't define its own).
    doc = first_line(cls.__doc__)
    if doc:
        lines.extend((doc, ""))

    paths = usage.get(cls, [])
    if paths:
        rendered_paths = ", ".join(f"`{p}`" for p in sorted(paths))
        lines.extend((f"**Appears at:** {rendered_paths}", ""))

    fields = cls.__pane_info__.fields
    if not fields:
        lines.extend(("_No fields._", ""))
        return lines

    field_docstrings = field_docs_all(cls)
    lines.extend(("| Field | Type | Required | Default | Notes |", "| --- | --- | --- | --- | --- |"))
    for field in fields:
        type_text = describe_field_type(field.type, link_dataclasses=class_names, hooks_prefix="../hooks/")
        required = "Yes" if not field.has_default() else "No"
        default_text = "—" if not field.has_default() else escape_cell(render_default(field.default))

        note_parts = []
        doc_text = field_docstrings.get(field.name)
        if doc_text:
            note_parts.append(escape_cell(doc_text))
        runtime_note = RUNTIME_DEFAULT_NOTES.get(f"{cls.__name__}.{field.name}")
        if runtime_note:
            note_parts.append(escape_cell(runtime_note))
        notes_text = " ".join(note_parts)

        lines.append(f"| `{field.out_name}` | {type_text} | {required} | {default_text} | {notes_text} |")
    lines.append("")
    return lines


def render_plan_reference(plan_dir: pathlib.Path, classes: list[type], usage: dict[type, list[str]]) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    class_names = {c.__name__ for c in classes}

    lines = [
        "# Plan reference",
        "",
        "<!-- Generated by scripts/generate_docs.py. Do not edit. -->",
        "",
        "This reference is generated from every Pane-backed plan class reachable "
        "from `phaser.plan.ReconsPlan` (see `phaser/plan.py`, `phaser/types.py`). "
        "`ReconsPlan` is listed first; every other class is sorted alphabetically "
        "by name. A field typed as a hook (built-in short name or external "
        "`\"package.module:function\"` reference) links to the corresponding page "
        "under the [hook reference](../hooks/index.md) instead of being expanded "
        "here — hook property schemas live there, not in this page.",
        "",
        "## Classes in this reference",
        "",
    ]
    for cls in classes:
        lines.append(f"- [`{cls.__name__}`](#{cls.__name__.lower()})")
    lines.append("")

    for cls in classes:
        lines.extend(render_plan_class(cls, usage, class_names))

    (plan_dir / "index.md").write_text("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Hook reference (docs/generated/hooks/*.md)
# ---------------------------------------------------------------------------


def parse_hook_entry(entry: tuple) -> tuple[str, t.Any, tuple]:
    ref, props_ty, *rest = entry
    deps = tuple(rest[0]) if rest else ()
    return ref, props_ty, deps


def discover_hook_classes() -> list[type]:
    seen: set[type] = set()
    stack: list[type] = [Hook]
    result: list[type] = []
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            result.append(sub)
            stack.append(sub)
    return sorted(result, key=lambda c: c.__name__)


def group_hook_entries(known: dict) -> list[tuple]:
    """Group built-in hook names by identical (ref, props type, dependencies),
    so aliases (multiple short names for the same target) render as one row."""
    groups: dict[tuple, list[str]] = {}
    for name in sorted(known.keys()):
        key = parse_hook_entry(known[name])
        groups.setdefault(key, []).append(name)
    rows = [(tuple(names), *key) for key, names in groups.items()]
    rows.sort(key=lambda row: row[0][0])
    return rows


GENERIC_HOOK_NOTE = (
    "Every hook family accepts either a built-in short name from the table "
    "below, or an external `\"package.module:function\"` reference. External "
    "hook properties are passed through unvalidated — `HookConverter.try_convert` "
    "(`phaser/hooks/hook.py`) only validates properties against a props class "
    "when the short name is registered in `known`."
)


def render_props_table(props_ty: t.Any, plan_class_names: t.Set[str]) -> list[str]:
    if is_dataclass_cls(props_ty) and props_ty.__name__ in plan_class_names:
        return [
            f"**Properties:** this hook's properties are a full plan class, "
            f"documented separately — see [`{props_ty.__name__}`](../plan/index.md) "
            "in the plan reference.",
            "",
        ]
    if not is_dataclass_cls(props_ty):
        type_text = describe_field_type(props_ty, link_dataclasses=None, hooks_prefix="")
        return [f"**Properties:** {type_text} — no fixed schema (accepts arbitrary properties).", ""]

    fields = props_ty.__pane_info__.fields
    if not fields:
        return ["**Properties:** none.", ""]

    docs = field_docs_all(props_ty)
    lines = ["| Property | Type | Required | Default | Description |", "| --- | --- | --- | --- | --- |"]
    for field in fields:
        type_text = describe_field_type(field.type, link_dataclasses=None, hooks_prefix="")
        required = "Yes" if not field.has_default() else "No"
        default_text = "—" if not field.has_default() else escape_cell(render_default(field.default))
        desc = escape_cell(docs.get(field.name, ""))
        lines.append(f"| `{field.out_name}` | {type_text} | {required} | {default_text} | {desc} |")
    lines.append("")
    return lines


def render_hook_family_page(hook_cls: type, plan_class_names: t.Set[str]) -> str:
    title = hook_family_name(hook_cls)
    lines = [
        f"# {title} hooks",
        "",
        "<!-- Generated by scripts/generate_docs.py. Do not edit. -->",
        "",
        f"Registry: `{hook_cls.__module__}.{hook_cls.__name__}.known`.",
        "",
        GENERIC_HOOK_NOTE,
        "",
    ]

    known = getattr(hook_cls, "known", {})
    entries = group_hook_entries(known)
    if not entries:
        lines.extend(("No built-in hooks are currently registered for this family.", ""))
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(("## Built-in hooks", ""))
    for names, ref, props_ty, deps in entries:
        heading = ", ".join(f"`{n}`" for n in names)
        lines.append(f"### {heading}")
        lines.append("")
        lines.append(f"- **Target:** `{ref}`")
        if deps:
            dep_texts = []
            for dep in deps:
                dep_obj = _DEPENDENCIES.get(dep)
                if dep_obj is None:
                    raise RuntimeError(
                        f"Hook '{ref}' declares unknown optional dependency {dep!r}; "
                        "add it to phaser/hooks/_dependencies.py or fix the "
                        "registration. This is a code defect, not something the "
                        "generator can guess past."
                    )
                dep_texts.append(f"`{dep}` ({dep_obj.install_instructions()})")
            lines.append(f"- **Optional dependencies:** {'; '.join(dep_texts)}")
        else:
            lines.append("- **Optional dependencies:** none declared")
        lines.append("")
        lines.extend(render_props_table(props_ty, plan_class_names))

    return "\n".join(lines).rstrip() + "\n"


def render_hooks_reference(hooks_dir: pathlib.Path, plan_class_names: t.Set[str]) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    families = discover_hook_classes()

    index_lines = [
        "# Hook reference",
        "",
        "<!-- Generated by scripts/generate_docs.py. Do not edit. -->",
        "",
        "One page per hook-family registry (a subclass of `phaser.hooks.hook.Hook` "
        "with a `known` dict of built-in short names), discovered by walking "
        "`Hook.__subclasses__()` after importing `phaser.plan`. Families are "
        "sorted alphabetically by class name.",
        "",
    ]
    for hook_cls in families:
        slug = hook_family_slug(hook_cls)
        index_lines.append(f"- [{hook_family_name(hook_cls)}]({slug}.md) (`{hook_cls.__name__}`)")
    index_lines.append("")
    (hooks_dir / "index.md").write_text("\n".join(index_lines).rstrip() + "\n")

    for hook_cls in families:
        slug = hook_family_slug(hook_cls)
        (hooks_dir / f"{slug}.md").write_text(render_hook_family_page(hook_cls, plan_class_names))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentation reference pages")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("docs/generated"),
        help="Generated documentation directory",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)

    api_dir = output / "api"
    api_dir.mkdir(parents=True)
    generate_state_reference(api_dir / "state.md")

    discovered, usage = discover_plan_graph(plan_module.ReconsPlan)
    ordered_classes = ordered_plan_classes(plan_module.ReconsPlan, discovered)
    plan_class_names = {c.__name__ for c in ordered_classes}

    render_plan_reference(output / "plan", ordered_classes, usage)
    render_hooks_reference(output / "hooks", plan_class_names)


if __name__ == "__main__":
    main()
