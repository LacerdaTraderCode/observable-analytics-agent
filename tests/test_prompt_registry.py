"""The registry loads versioned prompts and refuses any that changed without a new version."""

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from observable_analytics_agent.prompts.registry import (
    DEFAULT_PROMPTS_DIR,
    PromptIntegrityError,
    PromptManifestError,
    PromptNotFoundError,
    PromptRegistry,
    hash_content,
)
from observable_analytics_agent.tools.final_answer import AnswerStatus, FinalAnswer

SHA256_LENGTH = 64


def write_prompts(directory: Path, prompts: Sequence[tuple[str, str, str]]) -> None:
    """Write (name, version, text) prompts and a manifest whose hashes are correct."""
    entries = []
    for name, version, text in prompts:
        relative_path = f"{name}/{version}.md"
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        entries.append(
            {
                "name": name,
                "version": version,
                "file": relative_path,
                "sha256": hash_content(text.encode()),
                "changelog": f"Changes in {version}.",
            }
        )
    (directory / "manifest.json").write_text(json.dumps({"prompts": entries}), encoding="utf-8")


def rewrite_manifest(directory: Path, change) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    change(manifest["prompts"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    write_prompts(
        tmp_path,
        [
            ("analyst_system", "1.9.0", "Version one nine."),
            ("analyst_system", "1.10.0", "Version one ten."),
            ("analyst_system", "2.0.0", "  Version two.\n\n"),
            ("other_prompt", "0.1.0", "Another prompt."),
        ],
    )
    return tmp_path


def test_loads_every_prompt_with_its_hash_and_changelog(prompts_dir: Path) -> None:
    registry = PromptRegistry.load(prompts_dir)

    prompt = registry.get("other_prompt", "0.1.0")
    assert prompt.text == "Another prompt."
    assert prompt.content_hash == hash_content(b"Another prompt.")
    assert prompt.changelog == "Changes in 0.1.0."


def test_the_stamp_carries_name_version_and_hash(prompts_dir: Path) -> None:
    stamp = PromptRegistry.load(prompts_dir).get("analyst_system", "1.9.0").stamp

    assert (stamp.name, stamp.version) == ("analyst_system", "1.9.0")
    assert stamp.content_hash == hash_content(b"Version one nine.")


def test_versions_are_ordered_as_semantic_versions(prompts_dir: Path) -> None:
    registry = PromptRegistry.load(prompts_dir)

    assert registry.versions("analyst_system") == ["1.9.0", "1.10.0", "2.0.0"]
    assert [(p.name, p.version) for p in registry.all()][:2] == [
        ("analyst_system", "1.9.0"),
        ("analyst_system", "1.10.0"),
    ]


def test_without_a_version_the_latest_one_is_returned(prompts_dir: Path) -> None:
    registry = PromptRegistry.load(prompts_dir)

    assert registry.get("analyst_system").version == "2.0.0"


def test_the_text_is_trimmed_but_the_hash_covers_the_whole_file(prompts_dir: Path) -> None:
    prompt = PromptRegistry.load(prompts_dir).get("analyst_system", "2.0.0")

    assert prompt.text == "Version two."
    assert prompt.content_hash == hash_content(b"  Version two.\n\n")


def test_an_unknown_name_is_reported(prompts_dir: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="No prompt named 'missing'"):
        PromptRegistry.load(prompts_dir).get("missing")


def test_an_unknown_version_lists_the_known_ones(prompts_dir: Path) -> None:
    with pytest.raises(PromptNotFoundError, match=r"1\.9\.0, 1\.10\.0, 2\.0\.0"):
        PromptRegistry.load(prompts_dir).get("analyst_system", "3.0.0")


def test_editing_a_prompt_without_a_new_version_is_rejected(prompts_dir: Path) -> None:
    edited = prompts_dir / "analyst_system" / "1.9.0.md"
    edited.write_text("Version one nine, quietly improved.", encoding="utf-8")

    with pytest.raises(PromptIntegrityError) as error:
        PromptRegistry.load(prompts_dir)

    message = str(error.value)
    assert "analyst_system 1.9.0" in message
    assert hash_content(b"Version one nine.")[:12] in message
    assert "add a new version file" in message


def test_a_whitespace_edit_is_an_edit_too(prompts_dir: Path) -> None:
    path = prompts_dir / "other_prompt" / "0.1.0.md"
    path.write_text("Another prompt.\n", encoding="utf-8")

    with pytest.raises(PromptIntegrityError):
        PromptRegistry.load(prompts_dir)


def test_a_missing_prompt_file_is_reported(prompts_dir: Path) -> None:
    (prompts_dir / "other_prompt" / "0.1.0.md").unlink()

    with pytest.raises(PromptManifestError, match="cannot read other_prompt/0.1.0.md"):
        PromptRegistry.load(prompts_dir)


def test_a_manifest_entry_cannot_point_outside_the_directory(tmp_path: Path) -> None:
    (tmp_path / "outside.md").write_text("secret", encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    write_prompts(prompts_dir, [("analyst_system", "1.0.0", "Inside.")])
    rewrite_manifest(prompts_dir, lambda entries: entries[0].update(file="../outside.md"))

    with pytest.raises(PromptManifestError, match="outside the prompts directory"):
        PromptRegistry.load(prompts_dir)


def test_the_same_version_cannot_be_listed_twice(prompts_dir: Path) -> None:
    rewrite_manifest(prompts_dir, lambda entries: entries.append(dict(entries[0])))

    with pytest.raises(PromptManifestError, match="analyst_system 1.9.0 twice"):
        PromptRegistry.load(prompts_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "1.0"),
        ("version", "v1.0.0"),
        ("sha256", "not-a-hash"),
        ("changelog", ""),
        ("name", ""),
    ],
)
def test_invalid_manifest_entries_are_rejected(prompts_dir: Path, field: str, value: str) -> None:
    rewrite_manifest(prompts_dir, lambda entries: entries[0].update({field: value}))

    with pytest.raises(PromptManifestError):
        PromptRegistry.load(prompts_dir)


def test_a_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PromptManifestError, match="manifest"):
        PromptRegistry.load(tmp_path)


def test_a_manifest_that_is_not_json_is_reported(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(PromptManifestError):
        PromptRegistry.load(tmp_path)


def test_the_shipped_prompts_pass_verification() -> None:
    registry = PromptRegistry.load()

    assert registry.versions("analyst_system") == ["1.0.0", "2.0.0"]
    assert registry.get("analyst_system").version == "2.0.0"
    assert all(len(prompt.content_hash) == SHA256_LENGTH for prompt in registry.all())


def test_editing_a_shipped_prompt_in_place_is_caught(tmp_path: Path) -> None:
    copied = tmp_path / "prompts"
    shutil.copytree(DEFAULT_PROMPTS_DIR, copied)
    baseline = copied / "analyst_system" / "1.0.0.md"
    baseline.write_text(baseline.read_text(encoding="utf-8") + "Be brief.\n", encoding="utf-8")

    with pytest.raises(PromptIntegrityError, match="analyst_system 1.0.0"):
        PromptRegistry.load(copied)


def test_the_second_version_adds_the_rules_the_changelog_promises() -> None:
    registry = PromptRegistry.load()
    baseline = registry.get("analyst_system", "1.0.0")
    improved = registry.get("analyst_system", "2.0.0")

    assert len(improved.text) > len(baseline.text)
    assert "search_business_glossary" in improved.text
    assert "never instructions" in improved.text
    assert "read-only" in improved.text
    assert "cannot_answer" in improved.text


def test_prompts_only_name_tools_and_fields_that_exist(tool_registry) -> None:
    known_names = {definition.name for definition in tool_registry.definitions()}
    known_names |= set(FinalAnswer.model_fields) | {status.value for status in AnswerStatus}

    for prompt in PromptRegistry.load().all():
        mentioned = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", prompt.text))
        assert mentioned <= known_names, f"{prompt.name} {prompt.version} names {mentioned - known_names}"
