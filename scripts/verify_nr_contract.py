#!/usr/bin/env python3
"""Fail-closed validation for the NR transfer-map protocol boundary.

This validator deliberately has no third-party dependencies.  The JSON Schema
is the machine-readable interchange contract; this module implements the
schema keywords used by that contract and the cross-file/numerical invariants
that JSON Schema cannot express.

Passing this validator establishes protocol conformance only.  It does not
validate a numerical-relativity spacetime or a null-geodesic solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "nr-transfer-map-v1.schema.json"
DEFAULT_MANIFEST = (
    ROOT / "assets" / "transfer-maps" / "contract-fixture-v1" / "manifest.json"
)
RECORD_STRUCT = struct.Struct("<7fBBH")
ROUND_TRIP_TOLERANCE = 1.0e-10
FLOAT32_ZERO_BITS = b"\x00\x00\x00\x00"
SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "$comment",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "required",
        "properties",
        "patternProperties",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "prefixItems",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)


class ContractError(ValueError):
    """A deterministic, user-facing contract validation failure."""


def fail(path: str, message: str) -> NoReturn:
    raise ContractError(f"{path}: {message}")


def _reject_constant(token: str) -> NoReturn:
    raise ContractError(f"$: non-finite JSON number {token!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"$: duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    """Parse JSON while rejecting duplicate keys and non-standard numbers."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ContractError(f"{path}: unable to read JSON: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid JSON: {error}") from error


def audit_schema_dialect(schema: Any, path: str = "$schema") -> None:
    """Reject validation keywords that the zero-dependency evaluator ignores."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ContractError(f"{path}: a schema node must be an object or boolean")
    unknown = set(schema) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        raise ContractError(
            f"{path}: unsupported JSON Schema keyword(s): "
            + ", ".join(repr(keyword) for keyword in sorted(unknown))
        )

    for keyword in ("$defs", "properties", "patternProperties"):
        children = schema.get(keyword, {})
        if not isinstance(children, dict):
            raise ContractError(f"{path}.{keyword}: expected an object")
        for name, child in children.items():
            audit_schema_dialect(child, f"{path}.{keyword}.{name}")
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(keyword, [])
        if not isinstance(children, list):
            raise ContractError(f"{path}.{keyword}: expected an array")
        for index, child in enumerate(children):
            audit_schema_dialect(child, f"{path}.{keyword}[{index}]")
    for keyword in ("not", "if", "then", "else", "items", "additionalProperties"):
        if keyword in schema and isinstance(schema[keyword], (dict, bool)):
            audit_schema_dialect(schema[keyword], f"{path}.{keyword}")


def _json_equal(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return float(first) == float(second)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _json_equal(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _json_equal(first[key], second[key]) for key in first
        )
    return type(first) is type(second) and first == second


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ContractError(f"$schema: unsupported JSON Schema type {expected!r}")


def _resolve_local_ref(root_schema: dict[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise ContractError(
            f"$schema: only local JSON Schema references are supported, got {reference!r}"
        )
    value: Any = root_schema
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"$schema: unresolved reference {reference!r}")
        value = value[part]
    return value


def _schema_trial(
    value: Any,
    schema: Any,
    root_schema: dict[str, Any],
    path: str,
) -> ContractError | None:
    try:
        validate_json_schema(value, schema, root_schema, path)
    except ContractError as error:
        return error
    return None


def validate_json_schema(
    value: Any,
    schema: Any,
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    """Validate the Draft 2020-12 subset used by the protocol schema."""
    if schema is True:
        return
    if schema is False:
        fail(path, "value is forbidden by the schema")
    if not isinstance(schema, dict):
        raise ContractError("$schema: each schema node must be an object or boolean")

    if "$ref" in schema:
        referenced = _resolve_local_ref(root_schema, schema["$ref"])
        validate_json_schema(value, referenced, root_schema, path)

    if "allOf" in schema:
        for branch in schema["allOf"]:
            validate_json_schema(value, branch, root_schema, path)
    if "anyOf" in schema:
        failures = [
            error
            for branch in schema["anyOf"]
            if (error := _schema_trial(value, branch, root_schema, path)) is not None
        ]
        if len(failures) == len(schema["anyOf"]):
            fail(path, "value does not match any allowed schema branch")
    if "oneOf" in schema:
        matches = sum(
            _schema_trial(value, branch, root_schema, path) is None
            for branch in schema["oneOf"]
        )
        if matches != 1:
            fail(path, f"value must match exactly one schema branch, matched {matches}")
    if "not" in schema and _schema_trial(value, schema["not"], root_schema, path) is None:
        fail(path, "value matches a forbidden schema")
    if "if" in schema:
        condition_matches = (
            _schema_trial(value, schema["if"], root_schema, path) is None
        )
        selected = schema.get("then") if condition_matches else schema.get("else")
        if selected is not None:
            validate_json_schema(value, selected, root_schema, path)

    expected_type = schema.get("type")
    if expected_type is not None:
        alternatives = (
            [expected_type] if isinstance(expected_type, str) else expected_type
        )
        if not isinstance(alternatives, list) or not all(
            isinstance(item, str) for item in alternatives
        ):
            raise ContractError("$schema: type must be a string or string array")
        if not any(_schema_type_matches(value, item) for item in alternatives):
            fail(path, f"expected type {' or '.join(alternatives)}")

    if "const" in schema and not _json_equal(value, schema["const"]):
        fail(path, f"expected constant {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        fail(path, f"value {value!r} is not in the allowed enum")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                fail(path, f"missing required property {key!r}")

        properties = schema.get("properties", {})
        pattern_properties = schema.get("patternProperties", {})
        matched_keys: set[str] = set()
        for key, child_schema in properties.items():
            if key in value:
                validate_json_schema(
                    value[key],
                    child_schema,
                    root_schema,
                    f"{path}.{key}",
                )
                matched_keys.add(key)
        for pattern, child_schema in pattern_properties.items():
            compiled = re.compile(pattern)
            for key, item in value.items():
                if compiled.search(key):
                    validate_json_schema(
                        item,
                        child_schema,
                        root_schema,
                        f"{path}.{key}",
                    )
                    matched_keys.add(key)

        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in matched_keys:
                continue
            if additional is False:
                fail(path, f"unknown property {key!r}")
            if isinstance(additional, dict):
                validate_json_schema(
                    item,
                    additional,
                    root_schema,
                    f"{path}.{key}",
                )
        property_count = len(value)
        if property_count < schema.get("minProperties", 0):
            fail(path, "object has too few properties")
        if "maxProperties" in schema and property_count > schema["maxProperties"]:
            fail(path, "object has too many properties")

    if isinstance(value, list):
        count = len(value)
        if count < schema.get("minItems", 0):
            fail(path, "array has too few items")
        if "maxItems" in schema and count > schema["maxItems"]:
            fail(path, "array has too many items")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if any(_json_equal(item, earlier) for earlier in value[:index]):
                    fail(f"{path}[{index}]", "array item is not unique")

        prefix_items = schema.get("prefixItems", [])
        for index, child_schema in enumerate(prefix_items):
            if index < count:
                validate_json_schema(
                    value[index],
                    child_schema,
                    root_schema,
                    f"{path}[{index}]",
                )
        items_schema = schema.get("items", True)
        for index in range(len(prefix_items), count):
            validate_json_schema(
                value[index],
                items_schema,
                root_schema,
                f"{path}[{index}]",
            )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            fail(path, "string is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            fail(path, "string is longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            fail(path, f"string does not match pattern {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            fail(path, "number must be finite")
        if "minimum" in schema and number < schema["minimum"]:
            fail(path, f"number is below minimum {schema['minimum']}")
        if "maximum" in schema and number > schema["maximum"]:
            fail(path, f"number is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and number <= schema["exclusiveMinimum"]:
            fail(path, f"number must be greater than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and number >= schema["exclusiveMaximum"]:
            fail(path, f"number must be less than {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            divisor = float(schema["multipleOf"])
            quotient = number / divisor
            if not math.isclose(quotient, round(quotient), abs_tol=1.0e-12):
                fail(path, f"number is not a multiple of {divisor}")


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, "expected a JSON number, not a boolean or other type")
    result = float(value)
    if not math.isfinite(result):
        fail(path, "number must be finite")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "expected an integer, not a boolean or other type")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ContractError(f"{path}: unable to hash artifact: {error}") from error
    return digest.hexdigest()


def _safe_artifact_path(root: Path, relative: Any, path: str) -> Path:
    if not isinstance(relative, str) or not relative:
        fail(path, "artifact path must be a non-empty POSIX string")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative
    ):
        fail(path, "artifact path must be normalized, relative, and traversal-free")
    if "\\" in relative:
        fail(path, "artifact path must use POSIX '/' separators")

    candidate = root.joinpath(*pure.parts)
    cursor = root
    if cursor.is_symlink():
        fail(path, "a symlinked artifact root is forbidden")
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            fail(path, "symlinked artifacts or path components are forbidden")
    try:
        root_resolved = root.resolve(strict=True)
        candidate_resolved = candidate.resolve(strict=True)
    except OSError as error:
        fail(path, f"artifact does not resolve to an existing file: {error}")
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        fail(path, "artifact resolves outside the dataset root")
    if not candidate_resolved.is_file():
        fail(path, "artifact is not a regular file")
    return candidate_resolved


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(first, second))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _mat_mul(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> list[list[float]]:
    columns = list(zip(*second))
    return [[_dot(row, column) for column in columns] for row in first]


def _identity_error(matrix: Sequence[Sequence[float]]) -> float:
    return max(
        abs(value - (1.0 if row == column else 0.0))
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
    )


def _matrix(values: Any, dimension: int, path: str) -> list[list[float]]:
    if not isinstance(values, list) or len(values) != dimension:
        fail(path, f"expected a {dimension}x{dimension} matrix")
    result: list[list[float]] = []
    for row_index, row in enumerate(values):
        if not isinstance(row, list) or len(row) != dimension:
            fail(f"{path}[{row_index}]", f"expected {dimension} entries")
        result.append(
            [
                _finite_number(value, f"{path}[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            ]
        )
    return result


def _flat_matrix4(values: Any, path: str) -> list[list[float]]:
    vector = _vector(values, 16, path)
    return [vector[index : index + 4] for index in range(0, 16, 4)]


def _flat_matrix3(values: Any, path: str) -> list[list[float]]:
    vector = _vector(values, 9, path)
    return [vector[index : index + 3] for index in range(0, 9, 3)]


def _vector(values: Any, dimension: int, path: str) -> list[float]:
    if not isinstance(values, list) or len(values) != dimension:
        fail(path, f"expected a {dimension}-component vector")
    return [
        _finite_number(value, f"{path}[{index}]")
        for index, value in enumerate(values)
    ]


def _inverse(matrix: Sequence[Sequence[float]], path: str) -> list[list[float]]:
    dimension = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(dimension)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = max(range(column, dimension), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-15:
            fail(path, "matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[dimension:] for row in augmented]


def _max_vector_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second))


def _validate_hashed_artifact(
    artifact: dict[str, Any],
    root: Path,
    path: str,
) -> Path:
    resolved = _safe_artifact_path(root, artifact["uri"], f"{path}.uri")
    expected_size = _integer(artifact["byteLength"], f"{path}.byteLength")
    actual_size = resolved.stat().st_size
    if actual_size != expected_size:
        fail(
            f"{path}.byteLength",
            f"declares {expected_size} bytes but stored artifact has {actual_size}",
        )
    actual_hash = _sha256(resolved)
    if actual_hash != artifact["sha256"]:
        fail(
            f"{path}.sha256",
            f"hash mismatch: expected {artifact['sha256']}, got {actual_hash}",
        )
    return resolved


def _validate_manifest_sidecar(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    dataset_root = manifest_path.parent
    sidecar = _safe_artifact_path(
        dataset_root,
        manifest["integrity"]["manifestSidecar"],
        "$.integrity.manifestSidecar",
    )
    expected = f"{_sha256(manifest_path)}  manifest.json\n".encode("ascii")
    try:
        actual = sidecar.read_bytes()
    except OSError as error:
        raise ContractError(f"{sidecar}: unable to read manifest sidecar: {error}") from error
    if actual != expected:
        fail(
            "$.integrity.manifestSidecar",
            "sidecar must exactly match the lowercase SHA-256, two spaces, "
            "'manifest.json', and one newline",
        )


def _validate_provenance(
    manifest: dict[str, Any],
    schema_path: Path,
    manifest_path: Path,
) -> set[str]:
    provenance = manifest["provenance"]
    artifact_base = provenance["artifactUriBase"]
    if artifact_base == "repository-root":
        bundled_root = ROOT
    elif artifact_base == "manifest-directory":
        bundled_root = manifest_path.parent
    else:
        fail(
            "$.provenance.artifactUriBase",
            f"unsupported artifact URI base {artifact_base!r}",
        )

    seen_uris: set[str] = set()
    schema_artifacts: list[dict[str, Any]] = []
    generator_artifacts: list[dict[str, Any]] = []
    roles: set[str] = set()
    for index, artifact in enumerate(provenance["sourceArtifacts"]):
        path = f"$.provenance.sourceArtifacts[{index}]"
        uri = artifact["uri"]
        if uri in seen_uris:
            fail(f"{path}.uri", f"duplicate source artifact URI {uri!r}")
        seen_uris.add(uri)
        roles.add(artifact["role"])
        if artifact["storage"] == "bundled":
            _validate_hashed_artifact(artifact, bundled_root, path)
        elif artifact["storage"] == "external-reference":
            valid_doi = (
                re.fullmatch(r"(?:doi:)?10\.[0-9]{4,9}/\S+", uri) is not None
            )
            parsed = urlsplit(uri)
            valid_https = (
                parsed.scheme == "https"
                and bool(parsed.netloc)
                and parsed.username is None
                and parsed.password is None
                and not parsed.fragment
            )
            if not (valid_doi or valid_https):
                fail(
                    f"{path}.uri",
                    "external-reference artifacts must use an HTTPS URL or doi: URI",
                )
        else:
            fail(f"{path}.storage", f"unsupported storage {artifact['storage']!r}")
        if artifact["role"] == "schema":
            schema_artifacts.append(artifact)
        if artifact["role"] == "generator-source":
            generator_artifacts.append(artifact)

    if len(schema_artifacts) != 1:
        fail("$.provenance.sourceArtifacts", "missing the pinned schema artifact")
    schema_artifact = schema_artifacts[0]
    schema_size = schema_path.stat().st_size
    schema_hash = _sha256(schema_path)
    if (
        schema_artifact["byteLength"] != schema_size
        or schema_artifact["sha256"] != schema_hash
    ):
        fail(
            "$.provenance.sourceArtifacts",
            "the pinned schema artifact content does not match the schema used "
            "for validation",
        )
    if len(generator_artifacts) != 1:
        fail("$.provenance.sourceArtifacts", "missing the generator-source artifact")
    if provenance["generator"]["uri"] != generator_artifacts[0]["uri"]:
        fail(
            "$.provenance.generator.uri",
            "generator URI must exactly identify the generator-source artifact",
        )
    expected_revision = f"sha256:{generator_artifacts[0]['sha256']}"
    if provenance["generator"]["codeRevision"] != expected_revision:
        fail(
            "$.provenance.generator.codeRevision",
            "generator revision must exactly bind the generator-source SHA-256",
        )
    return roles


def _validate_measured_accuracy(manifest: dict[str, Any], path: str = "$.accuracy") -> None:
    accuracy = manifest["accuracy"]
    if accuracy["status"] != "measured":
        fail(f"{path}.status", "scientific datasets require measured accuracy")
    if accuracy["notMeasuredReason"] is not None:
        fail(f"{path}.notMeasuredReason", "must be null when accuracy is measured")
    for name in (
        "nrConvergence",
        "constraintNorms",
        "geodesicNullResidual",
        "interpolationError",
    ):
        section = accuracy[name]
        section_path = f"{path}.{name}"
        if section["status"] != "measured":
            fail(f"{section_path}.status", "accuracy section must be measured")
        if not isinstance(section["method"], str) or not section["method"].strip():
            fail(f"{section_path}.method", "measured section needs a method")
        if not isinstance(section["quantity"], str) or not section["quantity"].strip():
            fail(f"{section_path}.quantity", "measured section needs a quantity")
        _finite_number(section["value"], f"{section_path}.value")


def _validate_dataset_claims(
    manifest: dict[str, Any],
    artifact_roles: set[str],
) -> None:
    kind = manifest["datasetKind"]
    status = manifest["scientificStatus"]
    physical = manifest["physicalSystem"]
    simulation = manifest["provenance"]["sourceSimulation"]
    chart_status = manifest["coordinates"]["nrChart"]["status"]
    spacetime_mode = manifest["rayIntegration"]["spacetimeMode"]
    fixture_roles = {"near-zone-metric", "horizon-data"}
    boundary_role = manifest["escapeBoundary"]["referenceObserver"][
        "sourceArtifactRole"
    ]
    if boundary_role is not None and boundary_role not in artifact_roles:
        fail(
            "$.escapeBoundary.referenceObserver.sourceArtifactRole",
            f"source artifact role {boundary_role!r} is not present in provenance",
        )

    if kind == "synthetic-contract-fixture":
        for flag in (
            "sourceIsNumericalRelativity",
            "derivedFromNearZoneSpacetime",
            "derivedWithSlowLightGeodesics",
        ):
            if status[flag] is not False:
                fail(f"$.scientificStatus.{flag}", "fixture claim flag must be false")
        if manifest["renderable"]:
            fail("$.renderable", "contract fixture must not be renderable")
        if physical["kind"] != "synthetic-contract-fixture" or physical["vacuum"] is not None:
            fail("$.physicalSystem", "fixture must use the synthetic physical-system gate")
        for name in (
            "parameterEpochProtocolM",
            "massRatioQ",
            "eccentricity",
            "referenceOrbitalPhaseRad",
            "remnant",
        ):
            if physical[name] is not None:
                fail(
                    f"$.physicalSystem.{name}",
                    "fixture physical parameters must remain null",
                )
        if physical["dimensionlessSpins"]:
            fail(
                "$.physicalSystem.dimensionlessSpins",
                "fixture physical spins must remain empty",
            )
        if not isinstance(physical["notApplicableReason"], str) or not physical[
            "notApplicableReason"
        ].strip():
            fail(
                "$.physicalSystem.notApplicableReason",
                "fixture needs an explicit physical-parameter non-applicability reason",
            )
        if chart_status != "synthetic":
            fail("$.coordinates.nrChart.status", "fixture chart must be synthetic")
        if (
            simulation["kind"] != "none"
            or simulation["catalog"] is not None
            or simulation["doi"] is not None
            or simulation["evolutionCode"] is not None
        ):
            fail("$.provenance.sourceSimulation", "fixture must not claim an NR source")
        if spacetime_mode != "synthetic":
            fail("$.rayIntegration.spacetimeMode", "fixture integration must be synthetic")
        if artifact_roles & fixture_roles:
            fail(
                "$.provenance.sourceArtifacts",
                "fixture must not contain NR metric or horizon-data roles",
            )
        if manifest["accuracy"]["status"] != "not-measured":
            fail("$.accuracy.status", "fixture accuracy must remain not-measured")
        if manifest["accuracy"]["fixtureAssertions"] is None:
            fail("$.accuracy.fixtureAssertions", "fixture assertions are required")
        if any(
            value is not None
            for value in manifest["accuracy"]["outcomeFractions"].values()
        ):
            fail("$.accuracy.outcomeFractions", "fixture fractions must remain null")
        return

    if manifest["accuracy"]["fixtureAssertions"] is not None:
        fail("$.accuracy.fixtureAssertions", "scientific datasets cannot use fixture assertions")
    _validate_measured_accuracy(manifest)

    if kind == "nr-slow-light-transfer-map":
        for flag in (
            "sourceIsNumericalRelativity",
            "derivedFromNearZoneSpacetime",
            "derivedWithSlowLightGeodesics",
        ):
            if status[flag] is not True:
                fail(f"$.scientificStatus.{flag}", "NR transfer map requires this claim flag")
        if physical["kind"] != "binary-black-hole" or physical["vacuum"] is not True:
            fail("$.physicalSystem", "NR transfer map must declare a vacuum binary")
        for name in (
            "parameterEpochProtocolM",
            "massRatioQ",
            "eccentricity",
            "referenceOrbitalPhaseRad",
        ):
            _finite_number(physical[name], f"$.physicalSystem.{name}")
        if physical["massRatioQ"] < 1.0:
            fail("$.physicalSystem.massRatioQ", "binary mass ratio q must be at least one")
        if physical["eccentricity"] < 0.0:
            fail("$.physicalSystem.eccentricity", "binary eccentricity cannot be negative")
        if physical["notApplicableReason"] is not None:
            fail(
                "$.physicalSystem.notApplicableReason",
                "NR binary parameters are applicable, so this field must be null",
            )

        component_ids = physical["componentIds"]
        if len(component_ids) < 2:
            fail("$.physicalSystem.componentIds", "NR binary needs at least two components")
        spin_ids: list[str] = []
        for index, spin in enumerate(physical["dimensionlessSpins"]):
            spin_ids.append(spin["componentId"])
            vector = _vector(
                spin["vector"],
                3,
                f"$.physicalSystem.dimensionlessSpins[{index}].vector",
            )
            magnitude = math.sqrt(sum(component * component for component in vector))
            if magnitude > 1.0 + ROUND_TRIP_TOLERANCE:
                fail(
                    f"$.physicalSystem.dimensionlessSpins[{index}].vector",
                    f"dimensionless spin magnitude exceeds one ({magnitude:.9g})",
                )
        if len(spin_ids) != len(set(spin_ids)) or set(spin_ids) != set(component_ids):
            fail(
                "$.physicalSystem.dimensionlessSpins",
                "dimensionless spins must cover each physical component exactly once",
            )

        remnant = physical["remnant"]
        if not isinstance(remnant, dict):
            fail("$.physicalSystem.remnant", "NR binary needs remnant parameters")
        remnant_mass = _finite_number(
            remnant["massFraction"],
            "$.physicalSystem.remnant.massFraction",
        )
        if remnant_mass <= 0.0:
            fail("$.physicalSystem.remnant.massFraction", "remnant mass must be positive")
        remnant_spin = _vector(
            remnant["dimensionlessSpin"],
            3,
            "$.physicalSystem.remnant.dimensionlessSpin",
        )
        remnant_spin_magnitude = math.sqrt(
            sum(component * component for component in remnant_spin)
        )
        if remnant_spin_magnitude > 1.0 + ROUND_TRIP_TOLERANCE:
            fail(
                "$.physicalSystem.remnant.dimensionlessSpin",
                f"remnant dimensionless spin magnitude exceeds one "
                f"({remnant_spin_magnitude:.9g})",
            )
        if chart_status != "declared":
            fail("$.coordinates.nrChart.status", "NR chart must be declared")
        if simulation["kind"] != "catalog-simulation":
            fail("$.provenance.sourceSimulation.kind", "NR source must be a catalog simulation")
        for name in ("catalog", "doi"):
            if not isinstance(simulation[name], str) or not simulation[name].strip():
                fail(
                    f"$.provenance.sourceSimulation.{name}",
                    "NR source field must be non-empty",
                )
        evolution_code = simulation["evolutionCode"]
        if not isinstance(evolution_code, dict):
            fail(
                "$.provenance.sourceSimulation.evolutionCode",
                "NR source needs structured evolution-code provenance",
            )
        for name in ("name", "release"):
            if (
                not isinstance(evolution_code[name], str)
                or not evolution_code[name].strip()
            ):
                fail(
                    f"$.provenance.sourceSimulation.evolutionCode.{name}",
                    "NR evolution-code field must be non-empty",
                )
        commit = evolution_code["commit"]
        reason = evolution_code["commitNotAvailableReason"]
        if commit is not None:
            if not isinstance(commit, str) or not commit.strip():
                fail(
                    "$.provenance.sourceSimulation.evolutionCode.commit",
                    "evolution-code commit must be a non-empty string or null",
                )
            if reason is not None:
                fail(
                    "$.provenance.sourceSimulation.evolutionCode.commitNotAvailableReason",
                    "must be null when an evolution-code commit is supplied",
                )
        elif not isinstance(reason, str) or not reason.strip():
            fail(
                "$.provenance.sourceSimulation.evolutionCode.commitNotAvailableReason",
                "a missing evolution-code commit needs a non-empty reason",
            )
        if re.fullmatch(r"10\.[0-9]{4,9}/\S+", simulation["doi"]) is None:
            fail("$.provenance.sourceSimulation.doi", "NR source DOI is malformed")
        if not {"near-zone-metric", "horizon-data"} <= artifact_roles:
            fail(
                "$.provenance.sourceArtifacts",
                "NR transfer map requires near-zone-metric and horizon-data artifacts",
            )
        if spacetime_mode != "time-dependent":
            fail(
                "$.rayIntegration.spacetimeMode",
                "NR transfer map must use time-dependent slow-light integration",
            )
        for name in ("absolute", "relative", "nullConstraint"):
            value = manifest["rayIntegration"]["tolerances"][name]
            if value is None or _finite_number(
                value, f"$.rayIntegration.tolerances.{name}"
            ) <= 0.0:
                fail(
                    f"$.rayIntegration.tolerances.{name}",
                    "NR integration tolerance must be positive and measured",
                )
        if manifest["units"]["massNormalization"]["referenceEpochSourceM"] is None:
            fail(
                "$.units.massNormalization.referenceEpochSourceM",
                "NR mass normalization needs a source-coordinate reference epoch",
            )
        for index, target in enumerate(manifest["captureTargets"]):
            if target["sourceArtifactRole"] != "horizon-data":
                fail(
                    f"$.captureTargets[{index}].sourceArtifactRole",
                    "NR capture surfaces must be sourced from horizon-data",
                )
        return

    if kind == "stationary-reference-transfer-map":
        if status["sourceIsNumericalRelativity"] is not False:
            fail(
                "$.scientificStatus.sourceIsNumericalRelativity",
                "stationary reference is not an NR source",
            )
        if status["derivedFromNearZoneSpacetime"] is not False:
            fail(
                "$.scientificStatus.derivedFromNearZoneSpacetime",
                "stationary reference must not claim an NR near-zone source",
            )
        if status["derivedWithSlowLightGeodesics"] is not False:
            fail(
                "$.scientificStatus.derivedWithSlowLightGeodesics",
                "stationary reference must not claim slow-light derivation",
            )
        if physical["kind"] != "stationary-black-hole":
            fail("$.physicalSystem.kind", "stationary reference needs a stationary black hole")
        if simulation["kind"] != "stationary-reference":
            fail(
                "$.provenance.sourceSimulation.kind",
                "stationary dataset needs a stationary-reference source",
            )
        if chart_status != "declared" or spacetime_mode != "stationary":
            fail(
                "$.rayIntegration.spacetimeMode",
                "stationary reference needs a declared stationary spacetime",
            )
        return

    fail("$.datasetKind", f"unsupported dataset kind {kind!r}")


def _validate_outcome_reporting(
    manifest: dict[str, Any],
    record_total: int,
    counts: dict[str, int],
) -> None:
    if record_total <= 0:
        fail("$.chunks", "dataset contains no binary records")

    accuracy = manifest["accuracy"]
    fractions = accuracy["outcomeFractions"]
    actual = {
        name: counts[name] / record_total
        for name in (
            "escaped",
            "captured",
            "unresolved",
            "outside-domain",
            "integrator-failure",
            "missing",
        )
    }

    if not manifest["renderable"]:
        return
    if manifest["datasetKind"] == "synthetic-contract-fixture":
        fail("$.renderable", "synthetic contract fixtures cannot be rendered")
    _validate_measured_accuracy(manifest)
    if accuracy["fixtureAssertions"] is not None:
        fail(
            "$.accuracy.fixtureAssertions",
            "renderable datasets cannot use fixture assertions",
        )

    for name, expected in actual.items():
        declared = _finite_number(
            fractions[name],
            f"$.accuracy.outcomeFractions.{name}",
        )
        if not math.isclose(declared, expected, rel_tol=0.0, abs_tol=1.0e-12):
            fail(
                f"$.accuracy.outcomeFractions.{name}",
                f"declares {declared}, decoded {expected}",
            )

    unusable = sum(
        actual[name]
        for name in (
            "unresolved",
            "outside-domain",
            "integrator-failure",
            "missing",
        )
    )
    declared_unusable = _finite_number(
        fractions["unusable"],
        "$.accuracy.outcomeFractions.unusable",
    )
    if not math.isclose(
        declared_unusable,
        unusable,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        fail(
            "$.accuracy.outcomeFractions.unusable",
            f"declares {declared_unusable}, decoded {unusable}",
        )

    partition_sum = sum(actual.values())
    if not math.isclose(partition_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        fail(
            "$.accuracy.outcomeFractions",
            f"decoded ray outcomes do not form a partition (sum={partition_sum})",
        )
    if counts["escaped"] + counts["captured"] == 0:
        fail(
            "$.renderable",
            "renderable dataset has no resolved escaped or captured rays",
        )


def _validate_affine_matrix(matrix: Sequence[Sequence[float]], path: str) -> None:
    expected_last_row = [0.0, 0.0, 0.0, 1.0]
    if _max_vector_error(matrix[3], expected_last_row) >= ROUND_TRIP_TOLERANCE:
        fail(path, "matrix is not an affine homogeneous transform")
    _inverse(matrix, path)


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _validate_proper_rotation(
    matrix: Sequence[Sequence[float]],
    path: str,
) -> float:
    transpose = [list(column) for column in zip(*matrix)]
    orthogonality_error = _identity_error(_mat_mul(transpose, matrix))
    determinant = _determinant3(matrix)
    determinant_error = abs(determinant - 1.0)
    error = max(orthogonality_error, determinant_error)
    if error >= ROUND_TRIP_TOLERANCE:
        fail(
            path,
            "spatial transform must be a proper rotation "
            f"(det={determinant:.12g}, max error={error:.3e})",
        )
    return error


def _spatial_rotation(matrix4: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row[:3]) for row in matrix4[:3]]


def _validate_inverse_pair(
    forward: Sequence[Sequence[float]],
    backward: Sequence[Sequence[float]],
    path: str,
) -> float:
    _validate_affine_matrix(forward, f"{path}.forward")
    _validate_affine_matrix(backward, f"{path}.backward")
    forward_error = _identity_error(_mat_mul(forward, backward))
    backward_error = _identity_error(_mat_mul(backward, forward))
    error = max(forward_error, backward_error)
    if error >= ROUND_TRIP_TOLERANCE:
        fail(
            path,
            f"declared matrices are not mutual inverses; max error {error:.3e} "
            f"must be < {ROUND_TRIP_TOLERANCE:.1e}",
        )
    return error


def _transform(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    return _mat_vec(matrix, vector)


def _validate_point_round_trips(
    forward: Sequence[Sequence[float]],
    backward: Sequence[Sequence[float]],
    declared_tolerance: float,
    path: str,
) -> float:
    probes = (
        [0.0, 0.0, 0.0, 1.0],
        [1.25, -2.5, 0.75, 1.0],
        [-11.0, 7.0, 3.5, 1.0],
    )
    error = 0.0
    for probe in probes:
        recovered_a = _transform(backward, _transform(forward, probe))
        recovered_b = _transform(forward, _transform(backward, probe))
        error = max(
            error,
            _max_vector_error(recovered_a, probe),
            _max_vector_error(recovered_b, probe),
        )
    gate = min(ROUND_TRIP_TOLERANCE, declared_tolerance)
    if error >= gate:
        fail(
            path,
            f"point round-trip error {error:.3e} must be < {gate:.3e}",
        )
    return error


def _validate_tetrad(
    sample: dict[str, Any],
    sample_path: str,
) -> float:
    metric = _flat_matrix4(
        sample["metricCovariantNr"],
        f"{sample_path}.metricCovariantNr",
    )
    inverse_metric_declared = _flat_matrix4(
        sample["metricContravariantNr"],
        f"{sample_path}.metricContravariantNr",
    )
    four_velocity = _vector(
        sample["fourVelocityContravariantNr"],
        4,
        f"{sample_path}.fourVelocityContravariantNr",
    )
    tetrad = _matrix(
        sample["tetradContravariantNr"],
        4,
        f"{sample_path}.tetradContravariantNr",
    )

    symmetry_error = max(
        abs(metric[row][column] - metric[column][row])
        for row in range(4)
        for column in range(4)
    )
    if symmetry_error >= ROUND_TRIP_TOLERANCE:
        fail(
            f"{sample_path}.metricCovariantNr",
            f"metric symmetry error {symmetry_error:.3e} is too large",
        )
    inverse_metric = _inverse(metric, f"{sample_path}.metricCovariantNr")
    inverse_error = max(
        _identity_error(_mat_mul(metric, inverse_metric_declared)),
        _identity_error(_mat_mul(inverse_metric_declared, metric)),
        max(
            abs(inverse_metric[row][column] - inverse_metric_declared[row][column])
            for row in range(4)
            for column in range(4)
        ),
    )
    if inverse_error >= ROUND_TRIP_TOLERANCE:
        fail(
            f"{sample_path}.metricContravariantNr",
            f"declared inverse metric mismatch {inverse_error:.3e}",
        )
    if inverse_metric_declared[0][0] >= 0.0:
        fail(
            f"{sample_path}.metricContravariantNr[0]",
            f"inverse metric must have g^tt < 0, got "
            f"{inverse_metric_declared[0][0]:.9g}",
        )
    if four_velocity[0] <= 0.0:
        fail(f"{sample_path}.fourVelocityContravariantNr[0]", "observer is not future-directed")

    e0_error = _max_vector_error(tetrad[0], four_velocity)
    if e0_error >= ROUND_TRIP_TOLERANCE:
        fail(
            f"{sample_path}.tetradContravariantNr[0]",
            f"time leg must equal observer four-velocity; max error {e0_error:.3e}",
        )

    eta = [-1.0, 1.0, 1.0, 1.0]
    gram = [
        [_dot(tetrad[a], _mat_vec(metric, tetrad[b])) for b in range(4)]
        for a in range(4)
    ]
    orthonormality_error = max(
        abs(gram[a][b] - (eta[a] if a == b else 0.0))
        for a in range(4)
        for b in range(4)
    )
    if orthonormality_error >= ROUND_TRIP_TOLERANCE:
        fail(
            f"{sample_path}.tetradContravariantNr",
            f"tetrad is not orthonormal for signature -+++; max error "
            f"{orthonormality_error:.3e}",
        )

    round_trip_error = 0.0
    local_probes = (
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, -2.0, 0.5],
        [1.25, -0.75, 2.5, -4.0],
    )
    for local in local_probes:
        coordinate = [
            math.fsum(tetrad[a][mu] * local[a] for a in range(4))
            for mu in range(4)
        ]
        metric_coordinate = _mat_vec(metric, coordinate)
        recovered = [
            eta[a] * _dot(tetrad[a], metric_coordinate) for a in range(4)
        ]
        round_trip_error = max(
            round_trip_error,
            _max_vector_error(local, recovered),
        )
    if round_trip_error >= ROUND_TRIP_TOLERANCE:
        fail(
            f"{sample_path}.tetradContravariantNr",
            f"tetrad tangent-vector round-trip error {round_trip_error:.3e} "
            f"must be < {ROUND_TRIP_TOLERANCE:.1e}",
        )
    return max(
        symmetry_error,
        inverse_error,
        e0_error,
        orthonormality_error,
        round_trip_error,
    )


def _validate_frames_and_observers(manifest: dict[str, Any]) -> tuple[float, float]:
    assertions = manifest["accuracy"].get("fixtureAssertions")
    declared_tolerance = (
        _finite_number(
            assertions["worldCameraRoundTripTolerance"],
            "$.accuracy.fixtureAssertions.worldCameraRoundTripTolerance",
        )
        if isinstance(assertions, dict)
        else ROUND_TRIP_TOLERANCE
    )
    if declared_tolerance > ROUND_TRIP_TOLERANCE:
        fail(
            "$.accuracy.fixtureAssertions.worldCameraRoundTripTolerance",
            f"declared tolerance must be <= {ROUND_TRIP_TOLERANCE:.1e}",
        )

    world_frame = manifest["coordinates"]["worldFrame"]
    nr_to_world = _flat_matrix4(
        world_frame["nrToWorld"],
        "$.coordinates.worldFrame.nrToWorld",
    )
    world_to_nr = _flat_matrix4(
        world_frame["worldToNr"],
        "$.coordinates.worldFrame.worldToNr",
    )
    transform_error = _validate_inverse_pair(
        nr_to_world,
        world_to_nr,
        "$.coordinates.worldFrame",
    )
    transform_error = max(
        transform_error,
        _validate_proper_rotation(
            _spatial_rotation(nr_to_world),
            "$.coordinates.worldFrame.nrToWorld",
        ),
        _validate_proper_rotation(
            _spatial_rotation(world_to_nr),
            "$.coordinates.worldFrame.worldToNr",
        ),
    )
    transform_error = max(
        transform_error,
        _validate_point_round_trips(
            nr_to_world,
            world_to_nr,
            declared_tolerance,
            "$.coordinates.worldFrame",
        ),
    )
    sky = manifest["coordinates"]["sky"]
    if "worldToIcrs" in sky or "icrsToWorld" in sky:
        if "worldToIcrs" not in sky or "icrsToWorld" not in sky:
            fail(
                "$.coordinates.sky",
                "worldToIcrs and icrsToWorld rotations must be declared together",
            )
        world_to_icrs = _flat_matrix3(
            sky["worldToIcrs"],
            "$.coordinates.sky.worldToIcrs",
        )
        icrs_to_world = _flat_matrix3(
            sky["icrsToWorld"],
            "$.coordinates.sky.icrsToWorld",
        )
        sky_inverse_error = max(
            _identity_error(_mat_mul(world_to_icrs, icrs_to_world)),
            _identity_error(_mat_mul(icrs_to_world, world_to_icrs)),
        )
        if sky_inverse_error >= ROUND_TRIP_TOLERANCE:
            fail(
                "$.coordinates.sky",
                f"world/ICRS rotations are not mutual inverses; max error "
                f"{sky_inverse_error:.3e}",
            )
        transform_error = max(
            transform_error,
            sky_inverse_error,
            _validate_proper_rotation(
                world_to_icrs,
                "$.coordinates.sky.worldToIcrs",
            ),
            _validate_proper_rotation(
                icrs_to_world,
                "$.coordinates.sky.icrsToWorld",
            ),
        )

    camera = manifest["camera"]
    camera_to_world = _flat_matrix4(
        camera["cameraToWorld"],
        "$.camera.cameraToWorld",
    )
    world_to_camera = _flat_matrix4(
        camera["worldToCamera"],
        "$.camera.worldToCamera",
    )
    transform_error = max(
        transform_error,
        _validate_inverse_pair(
            camera_to_world,
            world_to_camera,
            "$.camera",
        ),
        _validate_point_round_trips(
            camera_to_world,
            world_to_camera,
            declared_tolerance,
            "$.camera",
        ),
    )
    transform_error = max(
        transform_error,
        _validate_proper_rotation(
            _spatial_rotation(camera_to_world),
            "$.camera.cameraToWorld",
        ),
        _validate_proper_rotation(
            _spatial_rotation(world_to_camera),
            "$.camera.worldToCamera",
        ),
    )

    observation_times = [
        _finite_number(value, f"$.sampling.observationTimesM[{index}]")
        for index, value in enumerate(manifest["sampling"]["observationTimesM"])
    ]
    if any(
        current <= previous
        for previous, current in zip(observation_times, observation_times[1:])
    ):
        fail(
            "$.sampling.observationTimesM",
            "observation times must be strictly increasing",
        )

    observer_samples = manifest["observer"]["samples"]
    if len(observer_samples) != len(observation_times):
        fail(
            "$.observer.samples",
            "there must be exactly one observer sample per observation time",
        )
    expected_indices = set(range(len(observation_times)))
    actual_indices = {
        _integer(sample["sampleIndex"], f"$.observer.samples[{index}].sampleIndex")
        for index, sample in enumerate(observer_samples)
    }
    if actual_indices != expected_indices or len(actual_indices) != len(observer_samples):
        fail(
            "$.observer.samples",
            "sampleIndex values must cover every observation time exactly once",
        )

    tetrad_error = 0.0
    sample_by_index: dict[int, dict[str, Any]] = {}
    source_zero = _finite_number(
        manifest["timeReference"]["sourceTimeAtProtocolZeroM"],
        "$.timeReference.sourceTimeAtProtocolZeroM",
    )
    for index, sample in enumerate(observer_samples):
        sample_path = f"$.observer.samples[{index}]"
        sample_index = sample["sampleIndex"]
        event = _vector(sample["eventNr"], 4, f"{sample_path}.eventNr")
        protocol_time = _finite_number(
            sample["protocolTimeM"],
            f"{sample_path}.protocolTimeM",
        )
        mapped_protocol_time = event[0] - source_zero
        if abs(mapped_protocol_time - protocol_time) >= ROUND_TRIP_TOLERANCE:
            fail(
                f"{sample_path}.eventNr[0]",
                "source event time does not map to protocolTimeM through "
                "timeReference",
            )
        if abs(protocol_time - observation_times[sample_index]) >= ROUND_TRIP_TOLERANCE:
            fail(
                f"{sample_path}.protocolTimeM",
                "observer protocolTimeM must equal its observationTimesM entry",
            )
        tetrad_error = max(tetrad_error, _validate_tetrad(sample, sample_path))
        sample_by_index[sample_index] = sample

    camera_origin_world = _transform(
        camera_to_world,
        [0.0, 0.0, 0.0, 1.0],
    )
    proper_times: list[float] = []
    for sample_index in range(len(observation_times)):
        sample = sample_by_index[sample_index]
        event = _vector(
            sample["eventNr"],
            4,
            f"$.observer.samples[{sample_index}].eventNr",
        )
        observer_world = _transform(
            nr_to_world,
            [event[1], event[2], event[3], 1.0],
        )
        origin_error = _max_vector_error(observer_world, camera_origin_world)
        if origin_error >= ROUND_TRIP_TOLERANCE:
            fail(
                f"$.observer.samples[{sample_index}].eventNr",
                "fixed camera is not anchored to this observer sample; "
                f"max error {origin_error:.3e}",
            )
        transform_error = max(transform_error, origin_error)
        proper_times.append(
            _finite_number(
                sample["properTimeM"],
                f"$.observer.samples[{sample_index}].properTimeM",
            )
        )
    if any(
        current <= previous
        for previous, current in zip(proper_times, proper_times[1:])
    ):
        fail(
            "$.observer.samples",
            "observer properTimeM must be strictly increasing",
        )
    return transform_error, tetrad_error


def _invalid_float_is_positive_zero(record: bytes, offset: int, components: int) -> bool:
    return all(
        record[offset + 4 * component : offset + 4 * (component + 1)]
        == FLOAT32_ZERO_BITS
        for component in range(components)
    )


def _validate_record(
    record: bytes,
    record_path: str,
    observation_time: float,
    outcome_names: dict[int, str],
    capture_targets: dict[int, dict[str, Any]],
    capture_none: int,
) -> str:
    if len(record) != RECORD_STRUCT.size:
        fail(record_path, "truncated binary record")
    (
        direction_x,
        direction_y,
        direction_z,
        frequency_shift,
        coordinate_lookback_time,
        null_residual,
        projection_error,
        outcome,
        capture_target,
        validity_mask,
    ) = RECORD_STRUCT.unpack(record)
    floats = (
        direction_x,
        direction_y,
        direction_z,
        frequency_shift,
        coordinate_lookback_time,
        null_residual,
        projection_error,
    )
    for index, value in enumerate(floats):
        if not math.isfinite(value):
            fail(f"{record_path}.float[{index}]", "binary record contains NaN or Infinity")

    if outcome not in outcome_names:
        fail(f"{record_path}.rayOutcome", f"unknown outcome code {outcome}")
    outcome_name = outcome_names[outcome]
    valid_direction = 1 << 0
    valid_frequency = 1 << 1
    valid_boundary = 1 << 2
    valid_null = 1 << 3
    valid_projection = 1 << 4
    known_bits = (
        valid_direction
        | valid_frequency
        | valid_boundary
        | valid_null
        | valid_projection
    )
    if validity_mask & ~known_bits:
        fail(
            f"{record_path}.validityMask",
            f"unknown validity bits 0x{validity_mask & ~known_bits:04x}",
        )

    expected_masks = {
        "escaped": known_bits,
        "captured": valid_boundary | valid_null | valid_projection,
        "unresolved": valid_null | valid_projection,
        "outside-domain": valid_boundary | valid_null | valid_projection,
        "integrator-failure": valid_null,
        "missing": 0,
    }
    expected_mask = expected_masks[outcome_name]
    if validity_mask != expected_mask:
        fail(
            f"{record_path}.validityMask",
            f"{outcome_name} requires mask 0x{expected_mask:04x}, "
            f"got 0x{validity_mask:04x}",
        )

    if outcome_name == "captured":
        if capture_target not in capture_targets:
            fail(
                f"{record_path}.captureTarget",
                f"captured ray references unknown target {capture_target}",
            )
        valid_from, valid_through = capture_targets[capture_target][
            "validityIntervalProtocolM"
        ]
        if not valid_from <= observation_time <= valid_through:
            fail(
                f"{record_path}.captureTarget",
                f"target {capture_target} is not valid at observation time "
                f"{observation_time}",
            )
    elif capture_target != capture_none:
        fail(
            f"{record_path}.captureTarget",
            f"{outcome_name} ray must use the no-target sentinel {capture_none}",
        )

    field_encodings = (
        ("escapeDirection", 0, 3, valid_direction),
        ("frequencyShiftG", 12, 1, valid_frequency),
        ("coordinateLookbackTimeM", 16, 1, valid_boundary),
        ("nullResidual", 20, 1, valid_null),
        ("projectionErrorPx", 24, 1, valid_projection),
    )
    for name, offset, components, bit in field_encodings:
        if not validity_mask & bit and not _invalid_float_is_positive_zero(
            record, offset, components
        ):
            fail(
                f"{record_path}.{name}",
                "invalid fields must use canonical positive float32 zero",
            )

    if validity_mask & valid_direction:
        norm = math.sqrt(
            direction_x * direction_x
            + direction_y * direction_y
            + direction_z * direction_z
        )
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            fail(
                f"{record_path}.escapeDirection",
                f"valid escape direction is not unit length (norm={norm:.9g})",
            )
    if validity_mask & valid_frequency and frequency_shift <= 0.0:
        fail(
            f"{record_path}.frequencyShiftG",
            "valid frequency shift g must be strictly positive",
        )
    if validity_mask & valid_boundary and coordinate_lookback_time < 0.0:
        fail(
            f"{record_path}.coordinateLookbackTimeM",
            "coordinate lookback time must be non-negative",
        )
    if validity_mask & valid_null and null_residual < 0.0:
        fail(f"{record_path}.nullResidual", "valid null residual must be non-negative")
    if validity_mask & valid_projection and projection_error < 0.0:
        fail(
            f"{record_path}.projectionErrorPx",
            "valid projection error must be non-negative",
        )
    return outcome_name


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        max(first_x, second_x) < min(first_x + first_width, second_x + second_width)
        and max(first_y, second_y)
        < min(first_y + first_height, second_y + second_height)
    )


def _validate_chunks_and_records(
    manifest: dict[str, Any],
    manifest_path: Path,
    artifact_roles: set[str],
) -> tuple[int, dict[str, int]]:
    dataset_root = manifest_path.parent
    width = _integer(manifest["projection"]["widthPixels"], "$.projection.widthPixels")
    height = _integer(
        manifest["projection"]["heightPixels"],
        "$.projection.heightPixels",
    )
    observation_times = [
        _finite_number(value, f"$.sampling.observationTimesM[{index}]")
        for index, value in enumerate(manifest["sampling"]["observationTimesM"])
    ]
    chunks = manifest["chunks"]
    canonical_keys: list[tuple[int, int, int]] = []
    rectangles_by_sample: dict[int, list[tuple[int, int, int, int]]] = {
        index: [] for index in range(len(observation_times))
    }
    seen_uris: set[str] = set()

    for index, chunk in enumerate(chunks):
        path = f"$.chunks[{index}]"
        sample_index = _integer(chunk["sampleIndex"], f"{path}.sampleIndex")
        if sample_index >= len(observation_times):
            fail(f"{path}.sampleIndex", "sample index is outside observationTimesM")
        tile = chunk["tile"]
        x = _integer(tile["x"], f"{path}.tile.x")
        y = _integer(tile["y"], f"{path}.tile.y")
        tile_width = _integer(tile["width"], f"{path}.tile.width")
        tile_height = _integer(tile["height"], f"{path}.tile.height")
        if x + tile_width > width or y + tile_height > height:
            fail(f"{path}.tile", "tile extends outside the declared image")
        canonical_keys.append((sample_index, y, x))
        rectangles_by_sample[sample_index].append((x, y, tile_width, tile_height))

        uri = chunk["uri"]
        if uri in seen_uris:
            fail(f"{path}.uri", f"duplicate chunk URI {uri!r}")
        seen_uris.add(uri)
        expected_records = tile_width * tile_height
        if chunk["recordCount"] != expected_records:
            fail(
                f"{path}.recordCount",
                f"tile requires {expected_records} records",
            )
        if chunk["recordBytes"] != RECORD_STRUCT.size:
            fail(
                f"{path}.recordBytes",
                f"record size must be {RECORD_STRUCT.size}",
            )
        if chunk["byteLength"] != expected_records * RECORD_STRUCT.size:
            fail(
                f"{path}.byteLength",
                "byte length must equal recordCount * recordBytes",
            )

    if canonical_keys != sorted(canonical_keys):
        fail(
            "$.chunks",
            "chunks are not in canonical (sampleIndex, tile.y, tile.x) order",
        )

    for sample_index, rectangles in rectangles_by_sample.items():
        if not rectangles:
            fail(
                "$.chunks",
                f"observation sample {sample_index} has no tile coverage",
            )
        for first_index, first in enumerate(rectangles):
            for second_index, second in enumerate(rectangles[first_index + 1 :], first_index + 1):
                if _rectangles_overlap(first, second):
                    fail(
                        "$.chunks",
                        f"tiles {first_index} and {second_index} overlap for "
                        f"sample {sample_index}",
                    )
        covered_area = sum(rectangle[2] * rectangle[3] for rectangle in rectangles)
        if covered_area != width * height:
            fail(
                "$.chunks",
                f"sample {sample_index} has a tile gap: covered {covered_area} of "
                f"{width * height} pixels",
            )

    layout = manifest["recordLayout"]
    outcome_names = {code: name for name, code in layout["rayOutcomes"].items()}
    if len(outcome_names) != len(layout["rayOutcomes"]):
        fail("$.recordLayout.rayOutcomes", "outcome codes must be unique")
    capture_none = layout["captureTargetNone"]
    capture_targets: dict[int, dict[str, Any]] = {}
    capture_ids: set[str] = set()
    capture_priorities: set[int] = set()
    physical_components = set(manifest["physicalSystem"]["componentIds"])
    for index, target in enumerate(manifest["captureTargets"]):
        code = target["code"]
        if code == capture_none:
            fail(
                f"$.captureTargets[{index}].code",
                "capture target collides with the no-target sentinel",
            )
        if code in capture_targets:
            fail(f"$.captureTargets[{index}].code", f"duplicate target code {code}")
        if target["id"] in capture_ids:
            fail(
                f"$.captureTargets[{index}].id",
                f"duplicate target id {target['id']!r}",
            )
        if target["id"] not in physical_components:
            fail(
                f"$.captureTargets[{index}].id",
                "capture target is not listed in physicalSystem.componentIds",
            )
        valid_from, valid_through = target["validityIntervalProtocolM"]
        if valid_from > valid_through:
            fail(
                f"$.captureTargets[{index}].validityIntervalProtocolM",
                "capture-target validity interval runs backward",
            )
        priority = target["classificationPriority"]
        if priority in capture_priorities:
            fail(
                f"$.captureTargets[{index}].classificationPriority",
                f"duplicate classification priority {priority}",
            )
        source_role = target["sourceArtifactRole"]
        if source_role is not None and source_role not in artifact_roles:
            fail(
                f"$.captureTargets[{index}].sourceArtifactRole",
                f"source artifact role {source_role!r} is not present in provenance",
            )
        capture_targets[code] = target
        capture_ids.add(target["id"])
        capture_priorities.add(priority)

    counts = {name: 0 for name in sorted(layout["rayOutcomes"])}
    record_total = 0
    for chunk_index, chunk in enumerate(chunks):
        path = f"$.chunks[{chunk_index}]"
        artifact = _validate_hashed_artifact(chunk, dataset_root, path)
        observation_time = observation_times[chunk["sampleIndex"]]
        try:
            with artifact.open("rb") as stream:
                for record_index in range(chunk["recordCount"]):
                    record = stream.read(RECORD_STRUCT.size)
                    outcome_name = _validate_record(
                        record,
                        f"{path}.records[{record_index}]",
                        observation_time,
                        outcome_names,
                        capture_targets,
                        capture_none,
                    )
                    counts[outcome_name] += 1
                    record_total += 1
                if stream.read(1):
                    fail(path, "chunk contains trailing bytes")
        except OSError as error:
            raise ContractError(f"{artifact}: unable to read chunk: {error}") from error

    fixture_assertions = manifest["accuracy"].get("fixtureAssertions")
    if isinstance(fixture_assertions, dict):
        expected_counts = fixture_assertions["expectedOutcomeCounts"]
        if counts != expected_counts:
            fail(
                "$.accuracy.fixtureAssertions.expectedOutcomeCounts",
                f"expected {expected_counts}, decoded {counts}",
            )
    unresolved_fraction = manifest["accuracy"]["unresolvedFraction"]
    if unresolved_fraction is not None:
        actual_fraction = counts["unresolved"] / record_total
        if not math.isclose(
            actual_fraction,
            float(unresolved_fraction),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            fail(
                "$.accuracy.unresolvedFraction",
                f"declares {unresolved_fraction}, decoded {actual_fraction}",
            )
    return record_total, counts


def validate_contract(
    manifest_path: Path = DEFAULT_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Validate a transfer-map dataset and return a deterministic report."""
    manifest_path = manifest_path.resolve()
    schema_path = schema_path.resolve()
    schema = load_json_strict(schema_path)
    if not isinstance(schema, dict):
        fail("$schema", "schema root must be an object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only JSON Schema Draft 2020-12 is supported")
    audit_schema_dialect(schema)
    manifest = load_json_strict(manifest_path)
    validate_json_schema(manifest, schema, schema)
    if not isinstance(manifest, dict):
        fail("$", "manifest root must be an object")
    if manifest_path.name != "manifest.json":
        fail("$", "v1 manifest file must be named 'manifest.json'")
    if RECORD_STRUCT.size != 32:
        raise ContractError("$internal: '<7fBBH' is not 32 bytes on this Python runtime")

    _validate_manifest_sidecar(manifest, manifest_path)
    artifact_roles = _validate_provenance(manifest, schema_path, manifest_path)
    _validate_dataset_claims(manifest, artifact_roles)
    transform_error, tetrad_error = _validate_frames_and_observers(manifest)
    record_total, counts = _validate_chunks_and_records(
        manifest,
        manifest_path,
        artifact_roles,
    )
    _validate_outcome_reporting(manifest, record_total, counts)

    return {
        "chunks": len(manifest["chunks"]),
        "id": manifest["id"],
        "maxFrameError": f"{transform_error:.3e}",
        "maxTetradError": f"{tetrad_error:.3e}",
        "outcomes": ", ".join(f"{name}:{counts[name]}" for name in sorted(counts)),
        "records": record_total,
        "schema": manifest.get("schema"),
        "status": "protocol-conformant",
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="path to an NR transfer-map manifest",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="path to the v1 JSON Schema",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_contract(arguments.manifest, arguments.schema)
    except ContractError as error:
        print(f"NR transfer-map contract validation failed: {error}", file=sys.stderr)
        return 1
    print("NR transfer-map contract checks passed")
    for key in sorted(report):
        print(f"  {key} = {report[key]}")
    print(
        "  scope = protocol conformance only; no NR spacetime or geodesic "
        "physics was validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
