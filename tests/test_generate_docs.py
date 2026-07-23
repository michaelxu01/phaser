"""
Tests for `scripts/generate_docs.py` (the documentation reference generator).

These tests run the generator as a subprocess (the same way a contributor or
CI would run it: `python scripts/generate_docs.py --output DIR`), so they
exercise exactly the code path documented in the authoring guide, and a few
unit tests against the module's pure helper functions.
"""

import filecmp
import importlib
import subprocess
import sys
import typing as t
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_docs.py"


def _run_generator(output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _iter_files(root: Path) -> t.List[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_generation_from_clean_directory_succeeds(tmp_path: Path):
    output = tmp_path / "generated"
    assert not output.exists()

    result = _run_generator(output)

    assert result.returncode == 0, result.stderr
    assert (output / "api" / "state.md").is_file()
    assert (output / "plan" / "index.md").is_file()
    assert (output / "hooks" / "index.md").is_file()
    # every discovered hook family produced its own page
    assert (output / "hooks" / "noise-model.md").is_file()
    assert (output / "hooks" / "engine.md").is_file()


def test_generation_overwrites_existing_directory(tmp_path: Path):
    output = tmp_path / "generated"
    output.mkdir()
    stale = output / "stale-leftover.md"
    stale.write_text("this should be removed by the generator\n")

    result = _run_generator(output)

    assert result.returncode == 0, result.stderr
    assert not stale.exists()


def test_deterministic_generation(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    result_a = _run_generator(out_a)
    result_b = _run_generator(out_b)

    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr

    files_a = _iter_files(out_a)
    files_b = _iter_files(out_b)
    assert files_a == files_b, "generated file trees differ in structure"

    match, mismatch, errors = filecmp.cmpfiles(out_a, out_b, files_a, shallow=False)
    assert not mismatch, f"non-deterministic generation in: {mismatch}"
    assert not errors, f"comparison errors for: {errors}"


def test_plan_reference_contains_known_fields(tmp_path: Path):
    output = tmp_path / "generated"
    result = _run_generator(output)
    assert result.returncode == 0, result.stderr

    text = (output / "plan" / "index.md").read_text()
    # freshness/sanity: fields that must exist somewhere in the plan schema
    assert "`niter`" in text
    assert "`buffer_n_groups`" in text, "buffer_n_groups missing from generated plan reference"
    assert "`shuffle_groups`" in text
    # the two concrete engine plans must both be documented
    assert "## `ConventionalEnginePlan`" in text
    assert "## `GradientEnginePlan`" in text
    # B14: sentinel default gets a curated runtime-resolution note
    assert "props.shuffle_groups or not props.compact" in text


def test_hooks_reference_contains_known_hooks(tmp_path: Path):
    output = tmp_path / "generated"
    result = _run_generator(output)
    assert result.returncode == 0, result.stderr

    noise_model_text = (output / "hooks" / "noise-model.md").read_text()
    assert "`poisson`" in noise_model_text

    solver_text = (output / "hooks" / "conventional-solver.md").read_text()
    assert "`lsqml`" in solver_text
    assert "`epie`" in solver_text

    index_text = (output / "hooks" / "index.md").read_text()
    assert "noise-model.md" in index_text
    assert "engine.md" in index_text


def test_state_reference_output_path_unchanged(tmp_path: Path):
    """Requirement: phaser.state generation keeps the same output path and
    still documents the public state classes."""
    output = tmp_path / "generated"
    result = _run_generator(output)
    assert result.returncode == 0, result.stderr

    text = (output / "api" / "state.md").read_text()
    assert "## `ReconsState`" in text
    assert "## `PreparedRecons`" in text


def test_hook_alias_grouping(tmp_path: Path):
    """Aliases (multiple short names for the same target) must be grouped
    into a single entry, not repeated."""
    output = tmp_path / "generated"
    result = _run_generator(output)
    assert result.returncode == 0, result.stderr

    text = (output / "hooks" / "cost-regularizer.md").read_text()
    assert "### `obj_tikh`, `obj_tikhonov`" in text


def test_engine_hook_redirects_to_plan_reference(tmp_path: Path):
    """The `engine` hook family's props are full plan classes, already
    documented in the plan reference — they must not be duplicated inline."""
    output = tmp_path / "generated"
    result = _run_generator(output)
    assert result.returncode == 0, result.stderr

    text = (output / "hooks" / "engine.md").read_text()
    assert "documented separately" in text
    assert "| `noise_model` |" not in text  # not an inline field table


# ---------------------------------------------------------------------------
# Unit tests against the generator's pure helper functions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_docs_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        module = importlib.import_module("generate_docs")
        yield module
    finally:
        sys.path.remove(str(SCRIPT.parent))


def test_hook_family_slug(generate_docs_module):
    m = generate_docs_module
    import phaser.hooks.solver as solver

    assert m.hook_family_slug(solver.NoiseModelHook) == "noise-model"
    assert m.hook_family_slug(solver.ConventionalSolverHook) == "conventional-solver"


def test_describe_field_type_optional(generate_docs_module):
    m = generate_docs_module
    import typing as t

    text = m.describe_field_type(t.Optional[int], link_dataclasses=None, hooks_prefix="")
    assert text == "`int` \\| `None`"


def test_describe_field_type_empty_dict(generate_docs_module):
    m = generate_docs_module
    from phaser.types import EmptyDict

    text = m.describe_field_type(EmptyDict, link_dataclasses=None, hooks_prefix="")
    assert text == "empty mapping `{}`"


def test_describe_field_type_links_known_dataclass(generate_docs_module):
    m = generate_docs_module
    import phaser.plan as plan_module

    text = m.describe_field_type(
        plan_module.InitPlan, link_dataclasses={"InitPlan"}, hooks_prefix="../hooks/"
    )
    assert text == "[`InitPlan`](#initplan)"


def test_discover_hook_classes_includes_known_families(generate_docs_module):
    m = generate_docs_module
    import phaser.plan  # noqa: F401  (ensures every registry module is imported)

    families = {cls.__name__ for cls in m.discover_hook_classes()}
    for expected in (
        "RawDataHook", "NoiseModelHook", "ConventionalSolverHook",
        "GradientSolverHook", "EngineHook", "CostRegularizerHook",
    ):
        assert expected in families


def test_unknown_optional_dependency_is_a_hard_error(generate_docs_module):
    """A hook declaring an optional dependency not present in
    `phaser/hooks/_dependencies.py` must fail generation loudly — this is a
    registration bug, and the generator must never guess install instructions
    for it."""
    m = generate_docs_module

    class FakeProps:
        pass

    class FakeFamilyHook:
        __module__ = "fake.module"
        known = {"fake_name": ("fake.module:fake_function", FakeProps, ("nonexistent-dependency",))}

    with pytest.raises(RuntimeError, match="nonexistent-dependency"):
        m.render_hook_family_page(FakeFamilyHook, plan_class_names=set())
