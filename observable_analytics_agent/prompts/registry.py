"""The prompt registry: loads prompts from files and refuses any that changed without a new version.

A prompt is a file under `prompts/`. The manifest declares, for every version, the file, a
changelog line and the SHA-256 of the file bytes. Loading recomputes the hash, so a prompt that
was edited in place fails loudly instead of silently changing what old traces and recordings mean.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from observable_analytics_agent.observability.records import PromptStamp

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MANIFEST_NAME = "manifest.json"
SEMANTIC_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class PromptError(Exception):
    """Base class for prompt registry failures."""


class PromptManifestError(PromptError):
    """The manifest, or a file it points to, cannot be used."""


class PromptIntegrityError(PromptError):
    """A prompt file no longer matches the hash declared for its version."""


class PromptNotFoundError(PromptError):
    """No prompt has the requested name or version."""


class PromptEntry(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    changelog: str = Field(min_length=1)


class PromptManifest(BaseModel):
    prompts: list[PromptEntry]


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    text: str
    content_hash: str
    changelog: str

    @property
    def stamp(self) -> PromptStamp:
        return PromptStamp(name=self.name, version=self.version, content_hash=self.content_hash)


def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class PromptRegistry:
    def __init__(self, prompts: Sequence[Prompt]) -> None:
        self._prompts = {(prompt.name, prompt.version): prompt for prompt in prompts}

    @classmethod
    def load(cls, directory: Path = DEFAULT_PROMPTS_DIR) -> Self:
        manifest = _read_manifest(directory)
        _reject_duplicates(manifest.prompts)
        return cls([_load_prompt(directory, entry) for entry in manifest.prompts])

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Return one version of a prompt, or the latest semantic version when none is given."""
        known_versions = self.versions(name)
        if not known_versions:
            raise PromptNotFoundError(f"No prompt named '{name}'.")
        chosen = version or known_versions[-1]
        if chosen not in known_versions:
            raise PromptNotFoundError(
                f"No version {chosen} of '{name}'. Versions: {', '.join(known_versions)}."
            )
        return self._prompts[(name, chosen)]

    def versions(self, name: str) -> list[str]:
        found = [version for prompt_name, version in self._prompts if prompt_name == name]
        return sorted(found, key=_version_key)

    def all(self) -> list[Prompt]:
        ordered = sorted(self._prompts, key=lambda key: (key[0], _version_key(key[1])))
        return [self._prompts[key] for key in ordered]


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _read_manifest(directory: Path) -> PromptManifest:
    path = directory / MANIFEST_NAME
    try:
        return PromptManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PromptManifestError(f"Cannot use the prompt manifest at {path}: {error}") from error


def _reject_duplicates(entries: Sequence[PromptEntry]) -> None:
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.name, entry.version)
        if key in seen:
            raise PromptManifestError(f"The manifest lists {entry.name} {entry.version} twice.")
        seen.add(key)


def _load_prompt(directory: Path, entry: PromptEntry) -> Prompt:
    path = (directory / entry.file).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise PromptManifestError(
            f"{entry.name} {entry.version}: {entry.file} is outside the prompts directory."
        )
    try:
        content = path.read_bytes()
    except OSError as error:
        raise PromptManifestError(
            f"{entry.name} {entry.version}: cannot read {entry.file}: {error}"
        ) from error
    actual_hash = hash_content(content)
    if actual_hash != entry.sha256:
        raise PromptIntegrityError(
            f"{entry.name} {entry.version}: {entry.file} no longer matches its declared hash "
            f"(declared {entry.sha256[:12]}, found {actual_hash[:12]}). A prompt cannot be edited "
            "in place: add a new version file and a manifest entry instead."
        )
    return Prompt(
        name=entry.name,
        version=entry.version,
        text=content.decode("utf-8").strip(),
        content_hash=actual_hash,
        changelog=entry.changelog,
    )
