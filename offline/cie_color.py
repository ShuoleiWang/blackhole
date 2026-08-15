"""Authenticated CIE 1931 true-colour integration for offline spectra.

The scientific boundary is explicit:

* the input is observer-frame ``I_nu`` on the exact 471-bin frequency grid
  corresponding to the official CIE 1931 2 degree table at 360--830 nm;
* ``I_nu`` is converted to ``I_lambda`` with the spectral-density Jacobian and
  integrated at 1 nm with the trapezoidal rule to produce unnormalised XYZ;
* XYZ-to-linear-sRGB is an unclamped linear coordinate transform;
* exposure, negative-channel clipping, tone mapping, gamut clipping and the
  sRGB transfer curve occur only in an explicitly requested display step.

UV, X-ray and other out-of-band bins are not false-coloured or folded into the
visible result.  An input grid containing extra, missing, reordered or rounded
bins fails closed.  CIE XYZ here is a standard-observer colourimetric
derivation, not a camera-sensor simulation or a prediction of absolute human
appearance in a viewing environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import struct
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CIE_CSV: Final = (
    ROOT / "assets" / "science" / "cie" / "CIE_xyz_1931_2deg.csv"
)
DEFAULT_CIE_METADATA: Final = (
    ROOT
    / "assets"
    / "science"
    / "cie"
    / "CIE_xyz_1931_2deg.csv_metadata.json"
)

LIGHT_SPEED_M_S: Final = 299_792_458.0
CIE_FIRST_WAVELENGTH_NM: Final = 360
CIE_LAST_WAVELENGTH_NM: Final = 830
CIE_WAVELENGTH_STEP_NM: Final = 1
CIE_ROW_COUNT: Final = 471
CIE_CSV_SHA256: Final = (
    "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"
)
CIE_CSV_MD5: Final = "17cca777db64b17170f06f67ce9d3ab7"
CIE_METADATA_SHA256: Final = (
    "03abcaecf4e63d77045ef57c4514b52c8bb1a46dd18e1f93a50044a0f4f481c8"
)
CIE_PARSED_BINARY64_SHA256: Final = (
    "53895edccfc58085088db80524be0220000fcff1efd8969254a229ac61de7eeb"
)
CIE_DATASET_DOI: Final = "10.25039/CIE.DS.xvudnb9b"
CIE_LICENSE_ID: Final = "CC BY-SA 4.0"
CIE_LICENSE_URI: Final = "https://creativecommons.org/licenses/by-sa/4.0/"
CIE_SAMPLE_ROW_ONE_BASED: Final = 120
CIE_SAMPLE_ROW_TEXT: Final = (
    "479,0.104297900000,0.1334528000000,0.856619300000"
)
CIE_COLUMN_SUMS: Final = (
    280245.0,
    106.865469489595,
    106.8569171011719,
    106.892251278636,
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "CIE 1931 2-degree standard-colorimetric-observer tristimulus "
            "derivation from observer-frame spectral radiance"
        ),
        "wavelengthRangeNm": [360, 830],
        "wavelengthIncrementNm": 1,
        "inputQuantity": "observer-frame spectral-specific-intensity-I_nu",
        "inputUnits": "W m^-2 sr^-1 Hz^-1",
        "outputUnits": "W m^-2 sr^-1 CIE tristimulus weighting",
        "integration": "I_nu-to-I_lambda Jacobian then 1-nm trapezoidal XYZ",
        "outOfBandPolicy": (
            "exact-visible-grid-only; UV, X-ray and other bins are rejected, "
            "not false-coloured"
        ),
        "isCameraSensorModel": False,
        "isAbsoluteHumanAppearanceModel": False,
        "isDisplayTransform": False,
        "prohibitedClaim": (
            "Do not describe standard-observer XYZ as a camera response, an "
            "absolute prediction of appearance, or a UV/X-ray false-colour map."
        ),
    }
)


class CieColorError(RuntimeError):
    """An authenticated CIE resource or colour conversion failed closed."""


def _fail(path: str, message: str) -> NoReturn:
    raise CieColorError(f"{path}: {message}")


def _read_stable_regular(path: Path, maximum_bytes: int, label: str) -> bytes:
    path = Path(path).absolute()
    descriptor: int | None = None
    try:
        if path.is_symlink():
            _fail(label, "symlinked resources are forbidden")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(label, "expected a regular file")
        if before.st_size > maximum_bytes:
            _fail(label, f"resource exceeds the {maximum_bytes}-byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    except CieColorError:
        raise
    except OSError as error:
        _fail(label, f"unable to read resource: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        _fail(label, "resource changed while it was being read")
    return payload


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(label, f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"invalid UTF-8 JSON: {error}")
    if not isinstance(result, dict):
        _fail(label, "metadata root must be an object")
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _finite_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(path, "expected a finite number")
    return float(value)


def _verify_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        identifier = _mapping(metadata["identifier"], "metadata.identifier")
        if identifier != {
            "identifier": CIE_DATASET_DOI,
            "identifierType": "DOI",
        }:
            _fail("metadata.identifier", "unexpected dataset DOI identity")
        if metadata["formats"] != ["text/csv"]:
            _fail("metadata.formats", "expected the official text/csv format")
        alternate = metadata["alternateIdentifiers"]
        if alternate != [
            {
                "alternateIdentifier": "CIE_xyz_1931_2deg.csv",
                "alternateIdentifierType": "fileName",
            }
        ]:
            _fail("metadata.alternateIdentifiers", "unexpected official file name")
        rights = metadata["rightsList"]
        if not isinstance(rights, list) or len(rights) != 1:
            _fail("metadata.rightsList", "expected one license declaration")
        right = _mapping(rights[0], "metadata.rightsList[0]")
        if (
            right.get("rightsIdentifier") != CIE_LICENSE_ID
            or right.get("rightsURI") != CIE_LICENSE_URI
        ):
            _fail("metadata.rightsList[0]", "unexpected dataset license")
        checksums = metadata["checksums"]
        if not isinstance(checksums, list):
            _fail("metadata.checksums", "expected checksum declarations")
        checksum_map = {
            str(entry["hashMethod"]): str(entry["checksum"])
            for entry in checksums
            if isinstance(entry, dict)
        }
        if checksum_map != {"md5": CIE_CSV_MD5, "sha256": CIE_CSV_SHA256}:
            _fail("metadata.checksums", "metadata does not bind the official CSV")
        if metadata.get("schemaName") != "CIEmetaDigitalProduct":
            _fail("metadata.schemaName", "unexpected CIE metadata schema")
        if metadata.get("schemaVersion") != 4:
            _fail("metadata.schemaVersion", "unexpected CIE metadata version")
        datatable = _mapping(metadata["datatableInfo"], "metadata.datatableInfo")
        if datatable.get("interpolationMethod") != "linear":
            _fail("metadata.datatableInfo.interpolationMethod", "must be linear")
        if datatable.get("extrapolationMethod") != "zero":
            _fail("metadata.datatableInfo.extrapolationMethod", "must be zero")
        headers = datatable["columnHeaders"]
        if not isinstance(headers, list) or len(headers) != 4:
            _fail("metadata.datatableInfo.columnHeaders", "expected four columns")
        expected_titles = (
            "lambda",
            "x_bar(lambda)",
            "y_bar(lambda)",
            "z_bar(lambda)",
        )
        for index, (header_value, title) in enumerate(zip(headers, expected_titles)):
            header = _mapping(
                header_value,
                f"metadata.datatableInfo.columnHeaders[{index}]",
            )
            expected_unit = "nm" if index == 0 else "dimensionless"
            expected_quantity = (
                "wavelength" if index == 0 else "colour-matching function"
            )
            if (
                header.get("title") != title
                or header.get("unit") != expected_unit
                or header.get("quantity") != expected_quantity
                or header.get("wavelength_first") != CIE_FIRST_WAVELENGTH_NM
                or header.get("wavelength_last") != CIE_LAST_WAVELENGTH_NM
                or header.get("wavelength_step") != CIE_WAVELENGTH_STEP_NM
            ):
                _fail(
                    f"metadata.datatableInfo.columnHeaders[{index}]",
                    "column identity or grid disagrees with the official table",
                )
        validations = datatable["validations"]
        if not isinstance(validations, list) or len(validations) != 2:
            _fail("metadata.datatableInfo.validations", "expected two validations")
        by_type = {
            entry.get("validationType"): entry
            for entry in validations
            if isinstance(entry, dict)
        }
        sums = _mapping(
            by_type.get("sumOfColumns"),
            "metadata.datatableInfo.validations.sumOfColumns",
        )
        parsed_sums = json.loads(sums.get("validationValue", ""))
        if tuple(parsed_sums) != CIE_COLUMN_SUMS:
            _fail(
                "metadata.datatableInfo.validations.sumOfColumns",
                "unexpected official column sums",
            )
        sample = _mapping(
            by_type.get("sampleRow"),
            "metadata.datatableInfo.validations.sampleRow",
        )
        if (
            sample.get("validationParameter") != str(CIE_SAMPLE_ROW_ONE_BASED)
            or sample.get("validationValue") != CIE_SAMPLE_ROW_TEXT
        ):
            _fail(
                "metadata.datatableInfo.validations.sampleRow",
                "unexpected official sample row",
            )
    except KeyError as error:
        _fail("metadata", f"missing required field {error.args[0]!r}")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        _fail("metadata", f"malformed official metadata: {error}")


@dataclass(frozen=True, slots=True)
class Cie1931Table:
    """Authenticated 1 nm CIE 1931 2 degree colour-matching functions."""

    wavelengths_nm: tuple[int, ...]
    x_bar: tuple[float, ...]
    y_bar: tuple[float, ...]
    z_bar: tuple[float, ...]
    csv_sha256: str = CIE_CSV_SHA256
    csv_md5: str = CIE_CSV_MD5
    metadata_sha256: str = CIE_METADATA_SHA256

    def __post_init__(self) -> None:
        expected_wavelengths = tuple(
            range(CIE_FIRST_WAVELENGTH_NM, CIE_LAST_WAVELENGTH_NM + 1)
        )
        if self.wavelengths_nm != expected_wavelengths:
            raise ValueError("CIE wavelength grid must be exactly 360--830 nm at 1 nm")
        channels = (self.x_bar, self.y_bar, self.z_bar)
        if any(len(channel) != CIE_ROW_COUNT for channel in channels):
            raise ValueError("CIE colour-matching channels must contain 471 rows")
        if any(
            not math.isfinite(value) or value < 0.0
            for channel in channels
            for value in channel
        ):
            raise ValueError("CIE colour-matching values must be finite and non-negative")
        if (
            self.csv_sha256 != CIE_CSV_SHA256
            or self.csv_md5 != CIE_CSV_MD5
            or self.metadata_sha256 != CIE_METADATA_SHA256
        ):
            raise ValueError("CIE table identity hashes are not the pinned official values")
        digest = hashlib.sha256()
        for row in zip(self.wavelengths_nm, *channels):
            digest.update(struct.pack("<I3d", *row))
        if digest.hexdigest() != CIE_PARSED_BINARY64_SHA256:
            raise ValueError("CIE binary64 table values are not the authenticated dataset")

    def descriptor(self) -> dict[str, Any]:
        return {
            "classification": SCIENTIFIC_STATUS["classification"],
            "columnSums": list(CIE_COLUMN_SUMS),
            "csvMd5": self.csv_md5,
            "csvSha256": self.csv_sha256,
            "datasetDoi": CIE_DATASET_DOI,
            "license": {
                "id": CIE_LICENSE_ID,
                "uri": CIE_LICENSE_URI,
            },
            "metadataSha256": self.metadata_sha256,
            "parsedBinary64Sha256": CIE_PARSED_BINARY64_SHA256,
            "observer": "CIE-1931-2-degree-standard-colorimetric-observer",
            "rowCount": CIE_ROW_COUNT,
            "sampleRow": {
                "oneBasedIndex": CIE_SAMPLE_ROW_ONE_BASED,
                "value": CIE_SAMPLE_ROW_TEXT,
            },
            "wavelengthGrid": {
                "firstNm": CIE_FIRST_WAVELENGTH_NM,
                "lastNm": CIE_LAST_WAVELENGTH_NM,
                "stepNm": CIE_WAVELENGTH_STEP_NM,
            },
        }


def _parse_csv(payload: bytes) -> Cie1931Table:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("CIE CSV", f"official table is not ASCII: {error}")
    lines = text.splitlines()
    if len(lines) != CIE_ROW_COUNT:
        _fail("CIE CSV", f"expected {CIE_ROW_COUNT} rows, found {len(lines)}")
    wavelengths: list[int] = []
    channels: tuple[list[float], list[float], list[float]] = ([], [], [])
    for index, line in enumerate(lines):
        columns = line.split(",")
        if len(columns) != 4:
            _fail(f"CIE CSV row {index + 1}", "expected exactly four columns")
        try:
            wavelength = int(columns[0])
        except ValueError:
            _fail(f"CIE CSV row {index + 1}", "wavelength must be an integer")
        expected = CIE_FIRST_WAVELENGTH_NM + index
        if wavelength != expected:
            _fail(
                f"CIE CSV row {index + 1}",
                f"expected wavelength {expected} nm, found {wavelength}",
            )
        values: list[float] = []
        for column_index, raw in enumerate(columns[1:], start=1):
            try:
                value = float(raw)
            except ValueError:
                _fail(
                    f"CIE CSV row {index + 1} column {column_index + 1}",
                    "expected a decimal number",
                )
            if not math.isfinite(value) or value < 0.0:
                _fail(
                    f"CIE CSV row {index + 1} column {column_index + 1}",
                    "colour-matching value must be finite and non-negative",
                )
            values.append(value)
        wavelengths.append(wavelength)
        for channel, value in zip(channels, values):
            channel.append(value)
    if lines[CIE_SAMPLE_ROW_ONE_BASED - 1] != CIE_SAMPLE_ROW_TEXT:
        _fail("CIE CSV sample row", "row 120 does not match official metadata")
    sums = (
        math.fsum(wavelengths),
        *(math.fsum(channel) for channel in channels),
    )
    for index, (actual, expected) in enumerate(zip(sums, CIE_COLUMN_SUMS)):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-13):
            _fail(
                f"CIE CSV column {index + 1}",
                f"sum {actual:.17g} disagrees with official {expected:.17g}",
            )
    return Cie1931Table(
        wavelengths_nm=tuple(wavelengths),
        x_bar=tuple(channels[0]),
        y_bar=tuple(channels[1]),
        z_bar=tuple(channels[2]),
    )


def load_authenticated_cie_1931_2deg(
    csv_path: Path | str = DEFAULT_CIE_CSV,
    metadata_path: Path | str = DEFAULT_CIE_METADATA,
) -> Cie1931Table:
    """Authenticate the exact official CIE resources and return their table."""

    csv_payload = _read_stable_regular(Path(csv_path), 64 * 1024, "CIE CSV")
    metadata_payload = _read_stable_regular(
        Path(metadata_path), 64 * 1024, "CIE metadata"
    )
    csv_sha256 = hashlib.sha256(csv_payload).hexdigest()
    csv_md5 = hashlib.md5(csv_payload).hexdigest()  # noqa: S324 - required source ID.
    metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
    if csv_sha256 != CIE_CSV_SHA256:
        _fail("CIE CSV", f"SHA-256 mismatch: {csv_sha256}")
    if csv_md5 != CIE_CSV_MD5:
        _fail("CIE CSV", f"MD5 mismatch: {csv_md5}")
    if metadata_sha256 != CIE_METADATA_SHA256:
        _fail("CIE metadata", f"SHA-256 mismatch: {metadata_sha256}")
    metadata = _strict_json(metadata_payload, "CIE metadata")
    _verify_metadata(metadata)
    table = _parse_csv(csv_payload)
    if (
        table.csv_sha256 != csv_sha256
        or table.csv_md5 != csv_md5
        or table.metadata_sha256 != metadata_sha256
    ):
        raise AssertionError("authenticated CIE table hashes drifted internally")
    return table


def cie_1931_frequency_grid_hz(
    table: Cie1931Table | None = None,
) -> tuple[float, ...]:
    """Return the exact product grid, strictly increasing in frequency.

    The CIE table is increasing in wavelength.  Scientific spectral products
    require increasing ``nu``, so the returned order corresponds to wavelengths
    830, 829, ..., 360 nm.  Consumers reverse intensities back to increasing
    wavelength before applying the trapezoidal rule.
    """

    if table is None:
        table = load_authenticated_cie_1931_2deg()
    if not isinstance(table, Cie1931Table):
        raise TypeError("table must be an authenticated Cie1931Table")
    frequencies = tuple(
        LIGHT_SPEED_M_S / (wavelength_nm * 1.0e-9)
        for wavelength_nm in reversed(table.wavelengths_nm)
    )
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise AssertionError("CIE frequency grid is not strictly increasing")
    return frequencies


@dataclass(frozen=True, slots=True)
class Cie1931Xyz:
    """Unnormalised scientific XYZ in integrated input-radiance units."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("scientific XYZ values must be finite and non-negative")

    @property
    def chromaticity_xy(self) -> tuple[float, float] | None:
        total = math.fsum((self.x, self.y, self.z))
        if total == 0.0:
            return None
        return self.x / total, self.y / total

    def descriptor(self) -> dict[str, Any]:
        return {
            "observer": "CIE-1931-2-degree-standard-colorimetric-observer",
            "quantity": "unnormalised-XYZ-from-integrated-observer-frame-I_nu",
            "units": "W m^-2 sr^-1 CIE tristimulus weighting",
            "normalizationApplied": False,
            "displayTransformApplied": False,
            "isCameraSensorModel": False,
            "isAbsoluteHumanAppearanceModel": False,
        }


def _finite_sequence(
    values: Sequence[float],
    label: str,
    *,
    non_negative: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        entries = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error
    result: list[float] = []
    for index, value in enumerate(entries):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label}[{index}] must be finite")
        normalized = float(value)
        if non_negative and normalized < 0.0:
            raise ValueError(f"{label}[{index}] must be non-negative")
        result.append(normalized)
    return tuple(result)


def spectral_i_nu_to_cie_xyz(
    observer_frequencies_hz: Sequence[float],
    specific_intensities_nu: Sequence[float],
    *,
    table: Cie1931Table | None = None,
) -> Cie1931Xyz:
    """Integrate exact-grid observer-frame ``I_nu`` to scientific CIE XYZ."""

    if table is None:
        table = load_authenticated_cie_1931_2deg()
    if not isinstance(table, Cie1931Table):
        raise TypeError("table must be an authenticated Cie1931Table")
    frequencies = _finite_sequence(
        observer_frequencies_hz,
        "observer_frequencies_hz",
        non_negative=False,
    )
    expected_frequencies = cie_1931_frequency_grid_hz(table)
    if len(frequencies) != CIE_ROW_COUNT:
        raise ValueError(
            "observer_frequencies_hz must contain exactly the 471 visible CIE bins"
        )
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer_frequencies_hz must be strictly increasing")
    if frequencies != expected_frequencies:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(frequencies, expected_frequencies)
            )
            if actual != expected
        )
        raise ValueError(
            "observer_frequencies_hz does not equal the exact CIE grid at "
            f"index {mismatch}; rounded or out-of-band bins are forbidden"
        )
    intensities = _finite_sequence(
        specific_intensities_nu,
        "specific_intensities_nu",
        non_negative=True,
    )
    if len(intensities) != CIE_ROW_COUNT:
        raise ValueError("specific_intensities_nu must contain exactly 471 values")

    # Frequency order is 830 -> 360 nm.  Restore the official increasing-
    # wavelength order before applying dnu/dlambda = -c/lambda^2.
    intensities_by_wavelength = tuple(reversed(intensities))
    channels: list[float] = []
    for colour_matching_function in (table.x_bar, table.y_bar, table.z_bar):
        terms: list[float] = []
        for index, (wavelength_nm, intensity_nu, response) in enumerate(
            zip(
                table.wavelengths_nm,
                intensities_by_wavelength,
                colour_matching_function,
            )
        ):
            wavelength_m = wavelength_nm * 1.0e-9
            endpoint_weight = 0.5 if index in (0, CIE_ROW_COUNT - 1) else 1.0
            # I_lambda is per metre.  Multiplying by 1 nm (1e-9 m) performs
            # the fixed-step trapezoidal integral in the same SI radiance scale.
            term = (
                endpoint_weight
                * intensity_nu
                * LIGHT_SPEED_M_S
                / (wavelength_m * wavelength_m)
                * response
                * 1.0e-9
            )
            if not math.isfinite(term) or term < 0.0:
                raise CieColorError("I_nu to I_lambda conversion overflowed")
            terms.append(term)
        integrated = math.fsum(terms)
        if not math.isfinite(integrated) or integrated < 0.0:
            raise CieColorError("CIE XYZ trapezoidal integration overflowed")
        channels.append(integrated)
    return Cie1931Xyz(*channels)


@dataclass(frozen=True, slots=True)
class LinearSrgb:
    """Unclamped scene-linear sRGB coordinates derived from scientific XYZ."""

    r: float
    g: float
    b: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in (self.r, self.g, self.b)):
            raise ValueError("linear sRGB values must be finite")

    def descriptor(self) -> dict[str, Any]:
        return {
            "colourspace": "linear-sRGB-D65",
            "clamped": False,
            "exposureApplied": False,
            "toneMappingApplied": False,
            "transferCurveApplied": False,
            "derivedFrom": "CIE-1931-2-degree-XYZ",
            "units": "same linear radiance scale as source XYZ",
        }


def cie_xyz_to_unclamped_linear_srgb(xyz: Cie1931Xyz) -> LinearSrgb:
    """Transform XYZ to linear sRGB without exposure, clamping or encoding."""

    if not isinstance(xyz, Cie1931Xyz):
        raise TypeError("xyz must be Cie1931Xyz")
    matrix = (
        (3.2409699419045226, -1.537383177570094, -0.4986107602930034),
        (-0.9692436362808796, 1.8759675015077202, 0.04155505740717559),
        (0.05563007969699366, -0.20397695888897652, 1.0569715142428786),
    )
    source = (xyz.x, xyz.y, xyz.z)
    values = tuple(
        math.fsum(matrix[row][column] * source[column] for column in range(3))
        for row in range(3)
    )
    if any(not math.isfinite(value) for value in values):
        raise CieColorError("XYZ to linear-sRGB transform overflowed")
    return LinearSrgb(*values)


@dataclass(frozen=True, slots=True)
class DisplaySrgb:
    """Explicitly display-derived, tone-mapped and sRGB-encoded triplet."""

    r: float
    g: float
    b: float
    exposure: float
    tone_mapper: str = "reinhard-rec709-luminance-uniform-gamut/v2"

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in (self.r, self.g, self.b)
        ):
            raise ValueError("display sRGB channels must lie in [0, 1]")
        if not math.isfinite(self.exposure) or self.exposure <= 0.0:
            raise ValueError("display exposure must be finite and positive")
        if self.tone_mapper != "reinhard-rec709-luminance-uniform-gamut/v2":
            raise ValueError("unsupported display tone mapper")

    def descriptor(self) -> dict[str, Any]:
        return {
            "colourspace": "sRGB-D65",
            "derivedDisplayOutput": True,
            "exposure": self.exposure,
            "negativeLinearPolicy": "clip-to-zero-at-display-boundary",
            "toneMappingDomain": "Rec.709-linear-luminance",
            "luminanceCoefficients": [0.2126, 0.7152, 0.0722],
            "toneMapper": self.tone_mapper,
            "mappedLuminance": "Y/(1+Y)",
            "uniformRgbToneScale": "mappedY/Y = 1/(1+Y)",
            "gamutPolicy": "uniform-max-channel-scale-if-needed",
            "nonNegativeLinearRgbRatioPreservedBeforeSrgbEncoding": True,
            "transferCurve": "IEC-61966-2-1-sRGB",
            "isScientificXyz": False,
        }


def _srgb_encode(linear: float) -> float:
    if linear <= 0.0031308:
        return 12.92 * linear
    return 1.055 * linear ** (1.0 / 2.4) - 0.055


def derive_display_srgb(
    linear_srgb: LinearSrgb,
    *,
    exposure: float,
) -> DisplaySrgb:
    """Explicitly derive bounded display sRGB from unclamped scene-linear RGB."""

    if not isinstance(linear_srgb, LinearSrgb):
        raise TypeError("linear_srgb must be LinearSrgb")
    if (
        isinstance(exposure, bool)
        or not isinstance(exposure, (int, float))
        or not math.isfinite(float(exposure))
        or exposure <= 0.0
    ):
        raise ValueError("exposure must be finite and positive")
    normalized_exposure = float(exposure)
    exposed_channels: list[float] = []
    for channel in (linear_srgb.r, linear_srgb.g, linear_srgb.b):
        exposed = channel * normalized_exposure
        if not math.isfinite(exposed):
            raise CieColorError("display exposure overflowed")
        exposed_channels.append(max(0.0, exposed))

    try:
        luminance = math.fsum(
            coefficient * channel
            for coefficient, channel in zip(
                (0.2126, 0.7152, 0.0722),
                exposed_channels,
            )
        )
    except OverflowError as error:
        raise CieColorError("display Rec.709 luminance overflowed") from error
    if not math.isfinite(luminance) or luminance < 0.0:
        raise CieColorError("display Rec.709 luminance is invalid")
    if luminance == 0.0:
        if any(channel > 0.0 for channel in exposed_channels):
            raise CieColorError(
                "positive display RGB has zero representable Rec.709 luminance"
            )
        tone_mapped_channels = (0.0, 0.0, 0.0)
    else:
        common_tone_scale = 1.0 / (1.0 + luminance)
        tone_mapped_channels = tuple(
            channel * common_tone_scale for channel in exposed_channels
        )
    maximum_channel = max(tone_mapped_channels)
    if maximum_channel > 1.0:
        display_linear_channels = tuple(
            channel / maximum_channel for channel in tone_mapped_channels
        )
    else:
        display_linear_channels = tone_mapped_channels

    encoded: list[float] = []
    for display_linear in display_linear_channels:
        if (
            not math.isfinite(display_linear)
            or display_linear < 0.0
            or display_linear > 1.0
        ):
            raise CieColorError("uniform display tone/gamut scaling failed")
        encoded_channel = _srgb_encode(display_linear)
        if not math.isfinite(encoded_channel):
            raise CieColorError("display sRGB encoding failed")
        encoded.append(min(1.0, max(0.0, encoded_channel)))
    return DisplaySrgb(*encoded, exposure=normalized_exposure)


__all__ = (
    "CIE_COLUMN_SUMS",
    "CIE_CSV_MD5",
    "CIE_CSV_SHA256",
    "CIE_DATASET_DOI",
    "CIE_LICENSE_ID",
    "CIE_METADATA_SHA256",
    "CIE_PARSED_BINARY64_SHA256",
    "CIE_ROW_COUNT",
    "Cie1931Table",
    "Cie1931Xyz",
    "CieColorError",
    "DEFAULT_CIE_CSV",
    "DEFAULT_CIE_METADATA",
    "DisplaySrgb",
    "LIGHT_SPEED_M_S",
    "LinearSrgb",
    "SCIENTIFIC_STATUS",
    "cie_1931_frequency_grid_hz",
    "cie_xyz_to_unclamped_linear_srgb",
    "derive_display_srgb",
    "load_authenticated_cie_1931_2deg",
    "spectral_i_nu_to_cie_xyz",
)
