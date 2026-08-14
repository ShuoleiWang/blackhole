"""Bounded, path-anchored authentication for immutable input artifacts.

The helpers in this module deliberately avoid pathname-based ``stat``/``open``
sequences.  Every existing path component is opened relative to a held parent
directory descriptor without following symbolic links, and the final regular
file is hashed through the same descriptor that was authenticated.  A second
component-by-component open proves that the public absolute path still names
the directories and file that supplied the digest.

These checks authenticate a stable finite file snapshot.  They do not turn a
``Path`` into a capability after this function returns; callers that consume
the bytes later must perform their own anchored read or reauthenticate them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Final, Sequence


AUTHENTICATED_ARTIFACT_READ_CHUNK_BYTES: Final = 1024 * 1024
MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH: Final = 16 * 1024 * 1024
MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH: Final = 64 * 1024 * 1024
MAXIMUM_AUTHENTICATED_LOCATOR_SYMLINK_COUNT: Final = 40
MAXIMUM_AUTHENTICATED_LOCATOR_TARGET_BYTE_LENGTH: Final = 4096
MAXIMUM_AUTHENTICATED_LOCATOR_TOTAL_TARGET_BYTES: Final = 64 * 1024
MAXIMUM_AUTHENTICATED_LOCATOR_COMPONENT_STEPS: Final = 4096

_PATH_TYPE: Final = type(Path())
_STABLE_FILE_FIELDS: Final = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class AuthenticatedArtifactError(RuntimeError):
    """Base class for fail-closed artifact authentication failures."""


class AuthenticatedArtifactChangedError(AuthenticatedArtifactError):
    """The opened file or its absolute path changed during authentication."""


class AuthenticatedArtifactResourceError(AuthenticatedArtifactError):
    """A declared file snapshot exceeds a fixed authentication budget."""


@dataclass(frozen=True, slots=True)
class AuthenticatedArtifactDigest:
    """Exact absolute path, stable byte length, and SHA-256 of one artifact."""

    path: Path
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not _PATH_TYPE or not self.path.is_absolute():
            raise TypeError("artifact digest path must be an exact absolute Path")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise TypeError("artifact digest byte_length must be exact and non-negative")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise TypeError("artifact digest sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class AuthenticatedArtifactLocator:
    """Stable binding from one lexical absolute locator to a canonical target."""

    locator_path: Path
    resolved_path: Path
    chain_binding_sha256: str

    def __post_init__(self) -> None:
        for path, name in (
            (self.locator_path, "locator_path"),
            (self.resolved_path, "resolved_path"),
        ):
            if type(path) is not _PATH_TYPE or not path.is_absolute():
                raise TypeError(f"artifact locator {name} must be an absolute Path")
        if (
            type(self.chain_binding_sha256) is not str
            or len(self.chain_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.chain_binding_sha256
            )
        ):
            raise TypeError("artifact locator binding must be lowercase SHA-256")


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        raise AuthenticatedArtifactError(
            f"platform lacks required secure artifact flag {name}"
        )
    return value


def _secure_open_flags() -> tuple[int, int]:
    """Return directory/final flags, failing before path I/O if unavailable."""

    no_follow = _required_open_flag("O_NOFOLLOW")
    directory = _required_open_flag("O_DIRECTORY")
    nonblock = _required_open_flag("O_NONBLOCK")
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    if os.open not in supports_dir_fd:
        raise AuthenticatedArtifactError(
            "platform lacks dir_fd support for secure artifact opens"
        )
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    if type(close_on_exec) is not int:
        raise AuthenticatedArtifactError("platform has an invalid O_CLOEXEC flag")
    directory_flags = os.O_RDONLY | no_follow | directory | close_on_exec
    file_flags = os.O_RDONLY | no_follow | nonblock | close_on_exec
    return directory_flags, file_flags


def _require_secure_locator_primitives() -> None:
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    if (
        os.stat not in supports_dir_fd
        or os.stat not in supports_follow_symlinks
        or os.readlink not in supports_dir_fd
    ):
        raise AuthenticatedArtifactError(
            "platform lacks secure dir_fd lstat/readlink locator primitives"
        )


def _validated_absolute_path(path: Path, label: str) -> tuple[str, ...]:
    if type(label) is not str or not label:
        raise TypeError("artifact label must be an exact non-empty string")
    if (
        type(path) is not _PATH_TYPE
        or not path.is_absolute()
        or path.anchor != os.sep
        or path == path.parent
    ):
        raise TypeError(f"{label} path must be an exact non-root absolute Path")
    components = path.parts[1:]
    if not components or any(
        type(component) is not str
        or component in ("", ".", "..")
        or os.sep in component
        for component in components
    ):
        raise AuthenticatedArtifactError(f"{label} path is not canonical")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if type(normalized) is not _PATH_TYPE or normalized != path:
        raise AuthenticatedArtifactError(f"{label} path is not canonical")
    return components


def _open_absolute_file_chain(
    path: Path,
    label: str,
) -> tuple[list[int], int]:
    """Open every component from ``/`` with no symlink traversal."""

    components = _validated_absolute_path(path, label)
    directory_flags, file_flags = _secure_open_flags()
    directories: list[int] = []
    file_descriptor = -1
    try:
        directories.append(os.open(os.sep, directory_flags))
        for component in components[:-1]:
            directories.append(
                os.open(
                    component,
                    directory_flags,
                    dir_fd=directories[-1],
                )
            )
        file_descriptor = os.open(
            components[-1],
            file_flags,
            dir_fd=directories[-1],
        )
        return directories, file_descriptor
    except OSError as error:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)
        raise AuthenticatedArtifactError(
            f"cannot open {label} without following symbolic links"
        ) from error


def _file_identity(snapshot: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return tuple(  # type: ignore[return-value]
        getattr(snapshot, field) for field in _STABLE_FILE_FIELDS
    )


def _path_chain_identity(
    directories: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode)
        for snapshot in (os.fstat(descriptor) for descriptor in directories)
    )


def _stat_document(
    path: Path,
    snapshot: os.stat_result,
    kind: str,
    *,
    symlink_target: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "ctimeNs": snapshot.st_ctime_ns,
        "device": snapshot.st_dev,
        "inode": snapshot.st_ino,
        "kind": kind,
        "mode": snapshot.st_mode,
        "mtimeNs": snapshot.st_mtime_ns,
        "pathBytesHex": os.fsencode(path).hex(),
        "size": snapshot.st_size,
    }
    if symlink_target is not None:
        result["symlinkTargetBytesHex"] = os.fsencode(symlink_target).hex()
    return result


def resolve_stable_artifact_locator(
    path: Path,
    label: str,
) -> AuthenticatedArtifactLocator:
    """Resolve an interpreter-owned locator while binding every symlink hop.

    Ordinary scientific inputs must call :func:`authenticate_stable_artifact`
    directly and therefore reject every symlink.  This separate resolver is
    only for runtime locators such as package-manager ``sys.executable`` paths.
    It brackets every ``readlinkat`` with full no-follow stat identities and
    returns a hash of the lexical locator, canonical target, directory chain,
    and raw link targets.  Callers must resolve again after hashing the target
    and require the returned object to be identical.
    """

    initial_components = list(_validated_absolute_path(path, label))
    directory_flags, _file_flags = _secure_open_flags()
    _require_secure_locator_primitives()
    pending = initial_components
    resolved_components: list[str] = []
    directories: list[int] = []
    documents: list[dict[str, object]] = []
    symlink_count = 0
    symlink_target_bytes = 0
    component_steps = 0
    try:
        directories.append(os.open(os.sep, directory_flags))
        while pending:
            component_steps += 1
            if component_steps > MAXIMUM_AUTHENTICATED_LOCATOR_COMPONENT_STEPS:
                raise AuthenticatedArtifactResourceError(
                    f"{label} locator exceeds the component-step limit"
                )
            component = pending.pop(0)
            if type(component) is not str or os.sep in component:
                raise AuthenticatedArtifactError(
                    f"{label} locator contains an invalid path component"
                )
            if component in ("", "."):
                continue
            if component == "..":
                if resolved_components:
                    resolved_components.pop()
                    os.close(directories.pop())
                continue

            parent_descriptor = directories[-1]
            try:
                before = os.stat(
                    component,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise AuthenticatedArtifactError(
                    f"cannot inspect {label} locator component"
                ) from error
            component_path = Path(os.sep, *resolved_components, component)
            if stat.S_ISLNK(before.st_mode):
                symlink_count += 1
                if symlink_count > MAXIMUM_AUTHENTICATED_LOCATOR_SYMLINK_COUNT:
                    raise AuthenticatedArtifactResourceError(
                        f"{label} locator exceeds the symlink-hop limit"
                    )
                try:
                    target = os.readlink(component, dir_fd=parent_descriptor)
                    after = os.stat(
                        component,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise AuthenticatedArtifactError(
                        f"cannot read {label} locator symlink"
                    ) from error
                if type(target) is not str or _file_identity(before) != _file_identity(
                    after
                ):
                    raise AuthenticatedArtifactChangedError(
                        f"{label} locator symlink changed while it was read"
                    )
                encoded_target = os.fsencode(target)
                if len(encoded_target) > MAXIMUM_AUTHENTICATED_LOCATOR_TARGET_BYTE_LENGTH:
                    raise AuthenticatedArtifactResourceError(
                        f"{label} locator symlink target exceeds its byte limit"
                    )
                symlink_target_bytes += len(encoded_target)
                if (
                    symlink_target_bytes
                    > MAXIMUM_AUTHENTICATED_LOCATOR_TOTAL_TARGET_BYTES
                ):
                    raise AuthenticatedArtifactResourceError(
                        f"{label} locator symlink targets exceed their total limit"
                    )
                documents.append(
                    _stat_document(
                        component_path,
                        after,
                        "symlink",
                        symlink_target=target,
                    )
                )
                target_path = Path(target)
                target_components = list(
                    target_path.parts[1:]
                    if target_path.is_absolute()
                    else target_path.parts
                )
                if target_path.is_absolute():
                    while len(directories) > 1:
                        os.close(directories.pop())
                    resolved_components.clear()
                pending = [*target_components, *pending]
                continue

            final = not pending
            if final:
                documents.append(_stat_document(component_path, before, "final"))
                resolved_components.append(component)
                break
            if not stat.S_ISDIR(before.st_mode):
                raise AuthenticatedArtifactError(
                    f"{label} locator has a non-directory ancestor"
                )
            try:
                opened = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
                opened_snapshot = os.fstat(opened)
            except OSError as error:
                raise AuthenticatedArtifactError(
                    f"cannot open {label} locator ancestor securely"
                ) from error
            if _file_identity(before) != _file_identity(opened_snapshot):
                os.close(opened)
                raise AuthenticatedArtifactChangedError(
                    f"{label} locator ancestor changed while it was opened"
                )
            documents.append(
                _stat_document(component_path, opened_snapshot, "directory")
            )
            directories.append(opened)
            resolved_components.append(component)

        resolved_path = Path(os.sep, *resolved_components)
        if resolved_path == resolved_path.parent:
            raise AuthenticatedArtifactError(
                f"{label} locator does not resolve to a non-root artifact"
            )
        binding_payload = (
            json.dumps(
                {
                    "components": documents,
                    "locatorPathBytesHex": os.fsencode(path).hex(),
                    "resolvedPathBytesHex": os.fsencode(resolved_path).hex(),
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        return AuthenticatedArtifactLocator(
            path,
            resolved_path,
            hashlib.sha256(binding_payload).hexdigest(),
        )
    finally:
        for descriptor in reversed(directories):
            os.close(descriptor)


def authenticate_stable_artifact(
    path: Path,
    label: str,
    *,
    maximum_byte_length: int = MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH,
    preceding_total_byte_length: int = 0,
    maximum_total_byte_length: int = MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH,
) -> AuthenticatedArtifactDigest:
    """Hash one stable regular file within fixed per-file and aggregate caps.

    The initial ``fstat`` size is checked against both caps before the first
    byte is read.  The loop retains a one-byte sentinel budget so concurrent
    growth cannot turn the authentication step into an unbounded stream.
    """

    for value, name in (
        (maximum_byte_length, "maximum_byte_length"),
        (preceding_total_byte_length, "preceding_total_byte_length"),
        (maximum_total_byte_length, "maximum_total_byte_length"),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{name} must be an exact non-negative integer")
    if preceding_total_byte_length > maximum_total_byte_length:
        raise AuthenticatedArtifactResourceError(
            "preceding artifact bytes exceed the fixed aggregate limit"
        )

    directories: list[int] = []
    descriptor = -1
    reopened_directories: list[int] = []
    reopened_descriptor = -1
    try:
        directories, descriptor = _open_absolute_file_chain(path, label)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthenticatedArtifactError(f"{label} must be a regular file")
        if before.st_size < 0:
            raise AuthenticatedArtifactError(f"{label} has an invalid size")
        if before.st_size > maximum_byte_length:
            raise AuthenticatedArtifactResourceError(
                f"{label} exceeds the fixed per-file byte limit"
            )
        if preceding_total_byte_length + before.st_size > maximum_total_byte_length:
            raise AuthenticatedArtifactResourceError(
                f"{label} exceeds the fixed aggregate byte limit"
            )

        digest = hashlib.sha256()
        byte_length = 0
        effective_limit = min(
            maximum_byte_length,
            maximum_total_byte_length - preceding_total_byte_length,
        )
        while True:
            remaining_with_sentinel = effective_limit - byte_length + 1
            block = os.read(
                descriptor,
                min(AUTHENTICATED_ARTIFACT_READ_CHUNK_BYTES, remaining_with_sentinel),
            )
            if not block:
                break
            byte_length += len(block)
            if byte_length > maximum_byte_length:
                raise AuthenticatedArtifactResourceError(
                    f"{label} grew beyond the fixed per-file byte limit"
                )
            if (
                preceding_total_byte_length + byte_length
                > maximum_total_byte_length
            ):
                raise AuthenticatedArtifactResourceError(
                    f"{label} grew beyond the fixed aggregate byte limit"
                )
            digest.update(block)

        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or _file_identity(before) != _file_identity(after)
            or byte_length != after.st_size
        ):
            raise AuthenticatedArtifactChangedError(
                f"{label} changed while it was authenticated"
            )

        original_path_identity = _path_chain_identity(directories)
        reopened_directories, reopened_descriptor = _open_absolute_file_chain(
            path,
            label,
        )
        reopened = os.fstat(reopened_descriptor)
        final = os.fstat(descriptor)
        if (
            _path_chain_identity(reopened_directories) != original_path_identity
            or _file_identity(reopened) != _file_identity(after)
            or _file_identity(final) != _file_identity(after)
        ):
            raise AuthenticatedArtifactChangedError(
                f"{label} path changed while it was authenticated"
            )
        return AuthenticatedArtifactDigest(path, byte_length, digest.hexdigest())
    except AuthenticatedArtifactError:
        raise
    except OSError as error:
        raise AuthenticatedArtifactError(f"cannot authenticate {label}") from error
    finally:
        if reopened_descriptor >= 0:
            os.close(reopened_descriptor)
        for current in reversed(reopened_directories):
            os.close(current)
        if descriptor >= 0:
            os.close(descriptor)
        for current in reversed(directories):
            os.close(current)


__all__ = [
    "AUTHENTICATED_ARTIFACT_READ_CHUNK_BYTES",
    "MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH",
    "MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH",
    "MAXIMUM_AUTHENTICATED_LOCATOR_COMPONENT_STEPS",
    "MAXIMUM_AUTHENTICATED_LOCATOR_SYMLINK_COUNT",
    "MAXIMUM_AUTHENTICATED_LOCATOR_TARGET_BYTE_LENGTH",
    "MAXIMUM_AUTHENTICATED_LOCATOR_TOTAL_TARGET_BYTES",
    "AuthenticatedArtifactChangedError",
    "AuthenticatedArtifactDigest",
    "AuthenticatedArtifactError",
    "AuthenticatedArtifactLocator",
    "AuthenticatedArtifactResourceError",
    "authenticate_stable_artifact",
    "resolve_stable_artifact_locator",
]
