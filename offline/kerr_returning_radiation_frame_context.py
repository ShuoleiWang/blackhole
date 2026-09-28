"""Authenticated frame input for returning-radiation thermal emission.

This module is a narrow trust boundary between the replay-certified returning
radiation thermal spectrum and a future finite-thickness frame/transfer layer.
It does not trace a camera ray and it does not modify the existing Novikov--
Thorne transfer.  Authentication performs one public, transitive same-code
replay of the supplied spectrum provider.  It then binds the provider and its
profile to the *original* finite-thickness surface owned by the underlying
full forward Kerr kernel.

For a production-forward cached source that replay scans authenticated cache
records exactly once and traces zero rays; direct sources still perform their
one full same-code kernel replay.  The exported snapshot contains immutable
emission inputs and provenance only;
it contains no live provider, profile, surface, or kernel.  Consequently
``ReturningThermalEmissionSnapshotV1.revalidate`` can authenticate its own
canonical tree after a pickle round-trip, but deliberately cannot establish
fresh physical provenance.  Only the process-local opaque authority can be
used to consume live objects without replaying the expensive ray kernel.

The emission closure is one face throughout.  The KERRBB-D20 angular proxy
``f(mu)=1/2+3 mu/4`` satisfies ``2 integral_0^1 mu f(mu) dmu=1``; there is no
implicit factor of two.  Per-annulus ``sigma T_eff^4 = F_out`` closure is
enforced with a fixed, non-relaxable binary64 relative cap.  This remains a
same-code authenticated input, not an independent oracle, artifact loader,
complete KERRBB model, returning-radiation stress/work term ``F_S``, spectral
redistribution calculation, scattering atmosphere, polarization model, or
GRMHD result.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import struct
import sys
import threading
from types import MappingProxyType, ModuleType
from typing import Any, Final, Mapping
import weakref

from offline.authenticated_artifact import (
    MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH,
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH,
    AuthenticatedArtifactChangedError,
    AuthenticatedArtifactError,
    authenticate_stable_artifact,
    resolve_stable_artifact_locator,
)
from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.kerr import KerrKerrSchildMetric
from offline.kerr_disk import (
    STEFAN_BOLTZMANN_W_M2_K4,
    StationaryNovikovThorneDisk,
    colour_corrected_planck_specific_intensity_nu,
)
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_returning_radiation_kernel import (
    KerrForwardReturningRadiationKernel,
    KerrForwardReturningRadiationKernelProjection,
)
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedReturningRadiationKernelExecution,
)
from offline.kerr_returning_radiation_kernel_jobs import FORWARD
from offline.kerr_returning_radiation_thermal_profile import (
    CERTIFIED_KERR_FORWARD_CLASSIFICATIONS,
    AxisymmetricReturningRadiationThermalProfile,
    _validated_profile_live_owners,
)
from offline.kerr_returning_radiation_thermal_spectrum import (
    CertifiedReturningRadiationThermalSpectrumProvider,
    verify_certified_returning_radiation_thermal_spectrum_provider,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-authenticated-frame-emission-context/v1"
)
D20_CERTIFICATE_IMPLEMENTATION_ID: Final = (
    f"{IMPLEMENTATION_ID}/d20-one-face-flux-certificate/v1"
)
SNAPSHOT_IMPLEMENTATION_ID: Final = (
    f"{IMPLEMENTATION_ID}/emission-snapshot/v1"
)

# This is deliberately not a caller option.  The current shared-code thermal
# construction closes at roughly a few tens of binary64 ulps; 2e-12 leaves
# room for its documented exp/log branch while remaining a hard scientific
# publication gate.
MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL: Final = 2.0e-12

# These are fixed authentication resource limits, not caller-tunable physics
# parameters.  A 4096-annulus four-face kernel already contains O(10^8)
# binary64 coefficients, so accepting a larger graph here would expose the
# replay boundary to unreasonable memory/time amplification.  The source
# closure is a small fixed production set; its separate caps keep snapshot
# revalidation and pre-replay file hashing bounded even under forged input.
MAXIMUM_AUTHENTICATED_ANNULUS_COUNT: Final = 4096
MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT: Final = 64
MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH: Final = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH
)
MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH: Final = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH
)
MAXIMUM_RUNTIME_ARTIFACT_BYTE_LENGTH: Final = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH
)
MAXIMUM_RUNTIME_ARTIFACT_TOTAL_BYTE_LENGTH: Final = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH
)

_D20_COEFFICIENT: Final = 1.5
_D20_NORMALIZATION: Final = 0.5
_NUMERIC_BACKEND_LOGICAL_PATH: Final = "runtime/numeric-backend.json"
_PATH_TYPE: Final = type(Path())

# Fixed transitive production closure for provider/profile/kernel replay and
# the identities frozen below.  Standard-library behavior is represented by
# the separately hashed numeric-runtime descriptor.
_SOURCE_CLOSURE_PATHS: Final = (
    "offline/__init__.py",
    "offline/authenticated_artifact.py",
    "offline/disk_atmosphere.py",
    "offline/geodesic.py",
    "offline/job.py",
    "offline/kerr.py",
    "offline/kerr_disk.py",
    "offline/kerr_finite_thickness.py",
    "offline/kerr_finite_thickness_area.py",
    "offline/kerr_finite_thickness_emitter.py",
    "offline/kerr_finite_thickness_launch.py",
    "offline/kerr_finite_thickness_surface.py",
    "offline/kerr_returning_radiation_frame_context.py",
    "offline/kerr_returning_radiation_kernel.py",
    "offline/kerr_returning_radiation_kernel_cached.py",
    "offline/kerr_returning_radiation_kernel_jobs.py",
    "offline/kerr_returning_radiation_rays.py",
    "offline/kerr_returning_radiation_receiver_kernel.py",
    "offline/kerr_returning_radiation_receiver_rays.py",
    "offline/kerr_returning_radiation_thermal_profile.py",
    "offline/kerr_returning_radiation_thermal_spectrum.py",
    "offline/novikov_thorne.py",
    "offline/radiative_transfer.py",
    "offline/returning_radiation.py",
    "offline/returning_radiation_fate_quadrature.py",
    "offline/spacetime.py",
)
_FROZEN_SOURCE_MODULE_FILE: Final = Path(os.path.abspath(__file__))
_FROZEN_SOURCE_ROOT: Final = _FROZEN_SOURCE_MODULE_FILE.parents[1]

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code authenticated returning-thermal frame input"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "requiredSurfaceType": "exact KerrFiniteThicknessMultiSurface",
        "requiredProviderType": (
            "exact CertifiedReturningRadiationThermalSpectrumProvider"
        ),
        "requiresSamePythonSurfaceObject": False,
        "bindsOriginalForwardKernelSurfaceByExactTree": True,
        "authenticationPerformsTransitiveProviderReplay": True,
        "fixedTransitiveSourceClosureBound": True,
        "sourceModuleOriginsRecheckedBeforeAuthentication": True,
        "sourceModuleExpectedPathsFrozenAtImport": True,
        "cachedAuthenticationRetracesRays": False,
        "authorityReplayOnRequire": False,
        "liveRevalidationOccursAtFrameBoundaries": True,
        "batchConsumesPrivateFrozenEmissionTable": True,
        "batchReadsLiveProviderOrKernel": False,
        "snapshotContainsLivePhysicsObjects": False,
        "snapshotContainsCachePathsOrExecution": False,
        "snapshotIsPicklePortableEvidence": True,
        "snapshotIsArtifactLoader": False,
        "angularLaw": "KERRBB D20 f(mu)=1/2+3mu/4",
        "angularFluxNormalization": "2 integral_0^1 mu f(mu) dmu = 1",
        "oneFaceEmission": True,
        "implicitFactorOfTwo": False,
        "maximumSigmaT4RelativeResidual": (
            MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL
        ),
        "maximumAuthenticatedAnnulusCount": (
            MAXIMUM_AUTHENTICATED_ANNULUS_COUNT
        ),
        "maximumSourceClosureEntryCount": (
            MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT
        ),
        "maximumSourceClosureEntryByteLength": (
            MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH
        ),
        "maximumSourceClosureTotalByteLength": (
            MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH
        ),
        "hasIndependentPhysicsOracle": False,
        "isCompleteKerrbb": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScatteringOrSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this process-local same-code authentication as "
            "an independent oracle, artifact loader, complete KERRBB, F_S, "
            "spectral redistribution, a solved atmosphere, polarization, or "
            "GRMHD."
        ),
    }
)


class KerrReturningRadiationFrameContextError(RuntimeError):
    """Base class for fail-closed frame-context authentication failures."""


class KerrReturningRadiationFrameContextVerificationError(
    KerrReturningRadiationFrameContextError
):
    """Raised when live or frozen emission evidence is inconsistent."""


def _source_module_name(logical_path: str) -> str:
    """Map one fixed closure path to its import-system owner."""

    if logical_path == "offline/__init__.py":
        return "offline"
    if (
        type(logical_path) is not str
        or not logical_path.startswith("offline/")
        or not logical_path.endswith(".py")
        or "/" in logical_path[len("offline/") : -len(".py")]
    ):
        raise KerrReturningRadiationFrameContextError(
            "source closure contains an unsupported Python module path"
        )
    return logical_path[: -len(".py")].replace("/", ".")


_SOURCE_CLOSURE_MODULE_OWNERS: Final = tuple(
    (
        _source_module_name(logical_path),
        sys.modules.get(_source_module_name(logical_path)),
        _FROZEN_SOURCE_ROOT / logical_path,
    )
    for logical_path in _SOURCE_CLOSURE_PATHS
)


def _require_exact_source_module_origins(
    *,
    _source_module_file: Path = _FROZEN_SOURCE_MODULE_FILE,
    _source_root: Path = _FROZEN_SOURCE_ROOT,
    _source_paths: tuple[str, ...] = _SOURCE_CLOSURE_PATHS,
    _module_owners: tuple[tuple[str, Any, Path], ...] = (
        _SOURCE_CLOSURE_MODULE_OWNERS
    ),
) -> Path:
    """Bind every executed closure module to the exact tree being hashed."""

    module_file = globals().get("__file__")
    if (
        type(module_file) is not str
        or Path(os.path.abspath(module_file)) != _source_module_file
        or _SOURCE_CLOSURE_PATHS is not _source_paths
    ):
        raise KerrReturningRadiationFrameContextError(
            "frame-context module or closure has no frozen source identity"
        )
    for logical_path, (module_name, expected_module, expected_file) in zip(
        _source_paths,
        _module_owners,
    ):
        module = sys.modules.get(module_name)
        if type(module) is not ModuleType or module is not expected_module:
            raise KerrReturningRadiationFrameContextError(
                f"source-closure module {module_name} is not the exact loaded module"
            )
        actual_file = getattr(module, "__file__", None)
        if (
            type(actual_file) is not str
            or Path(os.path.abspath(actual_file)) != expected_file
        ):
            raise KerrReturningRadiationFrameContextError(
                f"source-closure module {module_name} was loaded from another tree"
            )
    return _source_root


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationFrameContextError(
            "frame-context descriptor is not finite canonical JSON"
        ) from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _exact_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite exact float")
    return value


def _exact_positive_float(value: Any, label: str) -> float:
    result = _exact_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_nonnegative_float(value: Any, label: str) -> float:
    result = _exact_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _same_float(first: float, second: float) -> bool:
    return first.hex() == second.hex()


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare a primitive/dataclass tree without attacker equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_exact_tree(
                object.__getattribute__(actual, item.name),
                object.__getattribute__(expected, item.name),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
        return
    if type(expected) is float:
        differs = not math.isfinite(actual) or actual.hex() != expected.hex()
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{path} uses unsupported exact type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{path} differs from authenticated evidence"
        )


def _canonical_identity_tree(value: Any, path: str) -> Any:
    """Encode a live dataclass tree with exact binary64/type semantics."""

    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{path} contains a non-finite float"
            )
        return {"floatHex": value.hex()}
    if value_type is str:
        return {"str": value}
    if value_type is bool:
        return {"bool": value}
    if value_type is int:
        return {"intDecimal": str(value)}
    if value is None:
        return None
    if value_type is _PATH_TYPE:
        if not value.is_absolute():
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{path} contains a non-absolute cache path"
            )
        return {
            "absolutePath": value.as_posix(),
            "exactType": f"{value_type.__module__}.{value_type.__qualname__}",
        }
    if value_type is tuple:
        return [
            _canonical_identity_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "exactType": f"{value_type.__module__}.{value_type.__qualname__}",
            "fields": [
                [
                    item.name,
                    _canonical_identity_tree(
                        object.__getattribute__(value, item.name),
                        f"{path}.{item.name}",
                    ),
                ]
                for item in fields(value)
            ],
        }
    raise KerrReturningRadiationFrameContextVerificationError(
        f"{path} contains unsupported live type {value_type.__name__}"
    )


def _live_identity_sha256(value: Any, label: str) -> str:
    return _sha256_text(
        _canonical_json(_canonical_identity_tree(value, label))
    )


def _canonical_embedded_json(value: Any, label: str) -> tuple[str, Mapping[str, Any]]:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact canonical JSON string")
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} is malformed"
        ) from error
    if type(parsed) is not dict:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} must encode an object"
        )
    if _canonical_json(parsed).encode("utf-8") != value.encode("utf-8"):
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} is not canonical"
        )
    return value, parsed


@dataclass(frozen=True, slots=True)
class ReturningThermalSourceClosureEntryV1:
    """One path-, size-, and content-bound source/runtime artifact."""

    logical_path: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.logical_path) is not str or not self.logical_path:
            raise TypeError("source logical_path must be a non-empty exact str")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise TypeError("source byte_length must be a non-negative exact int")
        _digest(self.sha256, "source sha256")

    @property
    def binding_sha256(self) -> str:
        payload = (
            f"{self.logical_path}\0{self.byte_length}\0{self.sha256}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "bindingSha256": self.binding_sha256,
            "byteLength": self.byte_length,
            "logicalPath": self.logical_path,
            "sha256": self.sha256,
        }


def _validate_source_closure_resources(
    entries: tuple[ReturningThermalSourceClosureEntryV1, ...],
    label: str,
) -> None:
    """Revalidate every untrusted entry and enforce fixed byte/count caps."""

    if type(entries) is not tuple or not entries:
        raise TypeError(f"{label} must be a non-empty exact tuple")
    if len(entries) > MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} exceeds the fixed source-closure entry-count limit"
        )
    total_byte_length = 0
    for index, entry in enumerate(entries):
        if type(entry) is not ReturningThermalSourceClosureEntryV1:
            raise TypeError(f"{label}[{index}] has a non-exact entry type")
        # Frozen dataclasses can be forged through object.__new__ or a hostile
        # pickle.  Invoke the exact class validator again; do not dispatch via
        # the instance and do not trust that __post_init__ ran at construction.
        ReturningThermalSourceClosureEntryV1.__post_init__(entry)
        byte_length = object.__getattribute__(entry, "byte_length")
        if byte_length > MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH:
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{label}[{index}] exceeds the fixed per-entry byte limit"
            )
        total_byte_length += byte_length
        if total_byte_length > MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH:
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{label} exceeds the fixed total-byte limit"
            )


def _source_closure_manifest_sha256(
    entries: tuple[ReturningThermalSourceClosureEntryV1, ...],
) -> str:
    _validate_source_closure_resources(entries, "source closure")
    return _sha256_text(
        _canonical_json([entry.descriptor() for entry in entries])
    )


def _runtime_artifact_descriptor(
    path_value: Any,
    label: str,
    preceding_total_byte_length: int = 0,
) -> Mapping[str, Any]:
    if type(path_value) is not str or not path_value:
        raise KerrReturningRadiationFrameContextError(
            f"{label} runtime path is unavailable"
        )
    try:
        locator_path = Path(path_value)
        locator_before = resolve_stable_artifact_locator(
            locator_path,
            f"{label} runtime locator",
        )
        path = locator_before.resolved_path
        authenticated = authenticate_stable_artifact(
            path,
            f"{label} runtime artifact",
            maximum_byte_length=MAXIMUM_RUNTIME_ARTIFACT_BYTE_LENGTH,
            preceding_total_byte_length=preceding_total_byte_length,
            maximum_total_byte_length=MAXIMUM_RUNTIME_ARTIFACT_TOTAL_BYTE_LENGTH,
        )
        locator_after = resolve_stable_artifact_locator(
            locator_path,
            f"{label} runtime locator",
        )
        if locator_after != locator_before:
            raise AuthenticatedArtifactChangedError(
                f"{label} runtime locator changed while hashing its target"
            )
    except (AuthenticatedArtifactError, OSError, TypeError, ValueError) as error:
        raise KerrReturningRadiationFrameContextError(
            f"cannot authenticate the {label} runtime artifact"
        ) from error
    return {
        "artifactName": path.name,
        "byteLength": authenticated.byte_length,
        "locatorChainSha256": locator_before.chain_binding_sha256,
        "locatorPathSha256": hashlib.sha256(os.fsencode(locator_path)).hexdigest(),
        "resolvedPathSha256": hashlib.sha256(os.fsencode(path)).hexdigest(),
        "sha256": authenticated.sha256,
    }


def _numeric_backend_descriptor() -> Mapping[str, Any]:
    math_extension = getattr(math, "__file__", None)
    build_tag, build_date = platform.python_build()
    libc_name, libc_version = platform.libc_ver()
    math_artifact = _runtime_artifact_descriptor(
        math_extension,
        "math extension",
    )
    python_artifact = _runtime_artifact_descriptor(
        sys.executable,
        "Python executable",
        math_artifact["byteLength"],
    )
    return {
        "architectureBits": 8 * struct.calcsize("P"),
        "binary64MantissaBits": sys.float_info.mant_dig,
        "byteOrder": sys.byteorder,
        "floatRadix": sys.float_info.radix,
        "implementationId": "cpython-binary64-struct-libm/v2",
        "libc": {"implementation": libc_name, "version": libc_version},
        "machine": platform.machine(),
        "mathExtension": math_artifact,
        "operatingSystem": platform.system(),
        "operatingSystemRelease": platform.release(),
        "operatingSystemVersion": platform.version(),
        "processor": platform.processor(),
        "pythonBuild": {"date": build_date, "tag": build_tag},
        "pythonCacheTag": sys.implementation.cache_tag,
        "pythonCompiler": platform.python_compiler(),
        "pythonExecutable": python_artifact,
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "structDoubleBytes": struct.calcsize("d"),
    }


def _bounded_source_file_digest(
    path: Path,
    logical_path: str,
    preceding_total_byte_length: int,
) -> tuple[int, str]:
    """Authenticate one source through the shared anchored artifact reader."""

    try:
        authenticated = authenticate_stable_artifact(
            path,
            f"required source dependency {logical_path}",
            maximum_byte_length=MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH,
            preceding_total_byte_length=preceding_total_byte_length,
            maximum_total_byte_length=MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH,
        )
    except AuthenticatedArtifactChangedError as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"required source dependency changed while hashing: {logical_path}"
        ) from error
    except AuthenticatedArtifactError as error:
        raise KerrReturningRadiationFrameContextError(
            f"required source dependency is unavailable: {logical_path}"
        ) from error
    return authenticated.byte_length, authenticated.sha256


def _source_closure_manifest(
    byte_overrides: Mapping[str, bytes] | None = None,
    numeric_backend_override: Mapping[str, Any] | None = None,
) -> tuple[ReturningThermalSourceClosureEntryV1, ...]:
    """Read the fixed closure; overrides exist only for attack-oriented tests."""

    if byte_overrides is not None:
        if type(byte_overrides) is not dict or any(
            type(key) is not str or type(payload) is not bytes
            for key, payload in byte_overrides.items()
        ):
            raise TypeError("source byte overrides must be an exact str->bytes dict")
        unknown = set(byte_overrides) - set(_SOURCE_CLOSURE_PATHS)
        if unknown:
            raise ValueError("source byte override names an unknown dependency")
    if len(_SOURCE_CLOSURE_PATHS) + 1 > MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT:
        raise KerrReturningRadiationFrameContextError(
            "fixed source closure exceeds its entry-count resource limit"
        )
    root = _require_exact_source_module_origins()
    entries: list[ReturningThermalSourceClosureEntryV1] = []
    total_byte_length = 0
    for logical_path in _SOURCE_CLOSURE_PATHS:
        if byte_overrides is not None and logical_path in byte_overrides:
            payload = byte_overrides[logical_path]
            payload_byte_length = len(payload)
            if payload_byte_length > MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH:
                raise KerrReturningRadiationFrameContextError(
                    "required source override exceeds its fixed per-entry "
                    f"byte limit: {logical_path}"
                )
            if (
                total_byte_length + payload_byte_length
                > MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH
            ):
                raise KerrReturningRadiationFrameContextError(
                    "source overrides exceed the total-byte resource limit"
                )
            payload_sha256 = hashlib.sha256(payload).hexdigest()
        else:
            payload_byte_length, payload_sha256 = _bounded_source_file_digest(
                root / logical_path,
                logical_path,
                total_byte_length,
            )
        total_byte_length += payload_byte_length
        entries.append(
            ReturningThermalSourceClosureEntryV1(
                logical_path,
                payload_byte_length,
                payload_sha256,
            )
        )
    backend = (
        _numeric_backend_descriptor()
        if numeric_backend_override is None
        else numeric_backend_override
    )
    if type(backend) is not dict:
        raise TypeError("numeric backend override must be an exact dict")
    backend_payload = _canonical_json(backend).encode("utf-8")
    if len(backend_payload) > MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH:
        raise KerrReturningRadiationFrameContextError(
            "numeric-backend descriptor exceeds its per-entry byte limit"
        )
    total_byte_length += len(backend_payload)
    if total_byte_length > MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH:
        raise KerrReturningRadiationFrameContextError(
            "fixed source closure exceeds its total-byte resource limit"
        )
    entries.append(
        ReturningThermalSourceClosureEntryV1(
            _NUMERIC_BACKEND_LOGICAL_PATH,
            len(backend_payload),
            hashlib.sha256(backend_payload).hexdigest(),
        )
    )
    result = tuple(sorted(entries, key=lambda entry: entry.logical_path))
    if len({entry.logical_path for entry in result}) != len(result):
        raise KerrReturningRadiationFrameContextError(
            "source closure contains duplicate logical paths"
        )
    _validate_source_closure_resources(result, "source closure")
    return result


@dataclass(frozen=True, slots=True)
class ReturningThermalD20FluxCertificateV1:
    """Exact D20 angular normalization plus per-annulus F/T closure."""

    coefficient: float
    normalization: float
    hemisphere_flux_normalization: float
    minimum_intensity_multiplier: float
    maximum_intensity_multiplier: float
    sigma_t4_relative_residuals: tuple[float, ...]
    maximum_sigma_t4_relative_residual: float
    maximum_allowed_sigma_t4_relative_residual: float
    one_face: bool
    implicit_factor_of_two: bool
    _descriptor_json: str
    _descriptor_sha256: str

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        self.revalidate()
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        _validate_d20_certificate(self)


def _d20_certificate_descriptor(
    certificate: ReturningThermalD20FluxCertificateV1,
) -> Mapping[str, Any]:
    coefficient = _exact_float(certificate.coefficient, "D20 coefficient")
    normalization = _exact_float(
        certificate.normalization,
        "D20 normalization",
    )
    hemisphere = _exact_float(
        certificate.hemisphere_flux_normalization,
        "D20 hemisphere flux normalization",
    )
    minimum = _exact_positive_float(
        certificate.minimum_intensity_multiplier,
        "D20 minimum multiplier",
    )
    maximum = _exact_positive_float(
        certificate.maximum_intensity_multiplier,
        "D20 maximum multiplier",
    )
    residuals_raw = certificate.sigma_t4_relative_residuals
    if type(residuals_raw) is not tuple or not residuals_raw:
        raise TypeError("D20 sigma*T^4 residuals must be a non-empty exact tuple")
    residuals = tuple(
        _exact_nonnegative_float(value, f"D20 residual {index}")
        for index, value in enumerate(residuals_raw)
    )
    maximum_residual = _exact_nonnegative_float(
        certificate.maximum_sigma_t4_relative_residual,
        "maximum sigma*T^4 residual",
    )
    cap = _exact_positive_float(
        certificate.maximum_allowed_sigma_t4_relative_residual,
        "maximum allowed sigma*T^4 residual",
    )
    if not _same_float(coefficient, _D20_COEFFICIENT):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 coefficient must be exactly 1.5"
        )
    if not _same_float(normalization, _D20_NORMALIZATION):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 normalization must be exactly 0.5"
        )
    if not _same_float(hemisphere, 1.0):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 angular flux normalization does not close exactly"
        )
    if not _same_float(minimum, 0.5) or not _same_float(maximum, 1.25):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 endpoint multipliers are inconsistent"
        )
    if not _same_float(maximum_residual, max(residuals)):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 maximum sigma*T^4 residual is stale"
        )
    if not _same_float(cap, MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL):
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 residual cap is not the fixed non-relaxable value"
        )
    if maximum_residual > cap:
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 closure exceeds the non-relaxable relative cap"
        )
    if type(certificate.one_face) is not bool or certificate.one_face is not True:
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 certificate must retain one-face semantics"
        )
    if (
        type(certificate.implicit_factor_of_two) is not bool
        or certificate.implicit_factor_of_two is not False
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 certificate cannot introduce an implicit factor of two"
        )
    return {
        "angularLaw": {
            "coefficient": coefficient,
            "formula": "f(mu)=(1+b*mu)/(1+2*b/3)=1/2+3*mu/4",
            "hemisphereFluxNormalization": hemisphere,
            "maximumIntensityMultiplier": maximum,
            "minimumIntensityMultiplier": minimum,
            "normalization": normalization,
            "normalizationEquation": "2 integral_0^1 mu f(mu) dmu = 1",
        },
        "fluxTemperatureClosure": {
            "equation": "sigma*T_eff^4=F_out",
            "maximumAllowedRelativeResidual": cap,
            "maximumRelativeResidual": maximum_residual,
            "relativeResiduals": residuals,
            "stefanBoltzmannWM2K4": STEFAN_BOLTZMANN_W_M2_K4,
        },
        "implementationId": D20_CERTIFICATE_IMPLEMENTATION_ID,
        "surfaceSemantics": {
            "implicitFactorOfTwo": False,
            "oneFace": True,
        },
    }


def _validate_d20_certificate(
    certificate: ReturningThermalD20FluxCertificateV1,
) -> None:
    if type(certificate) is not ReturningThermalD20FluxCertificateV1:
        raise TypeError("D20 certificate must have its exact public type")
    descriptor = _d20_certificate_descriptor(certificate)
    expected_json = _canonical_json(descriptor)
    supplied_json = object.__getattribute__(certificate, "_descriptor_json")
    supplied_sha = object.__getattribute__(certificate, "_descriptor_sha256")
    if type(supplied_json) is not str or type(supplied_sha) is not str:
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 descriptor identity has non-exact types"
        )
    if (
        supplied_json.encode("utf-8") != expected_json.encode("utf-8")
        or supplied_sha.encode("ascii")
        != _sha256_text(expected_json).encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 certificate differs from its canonical tree"
        )


def _sigma_t4_relative_residual(flux_value: Any, temperature_value: Any) -> float:
    """Close one annulus, preserving exact positive-sign zero semantics."""

    flux = _exact_nonnegative_float(flux_value, "outgoing flux")
    temperature = _exact_nonnegative_float(
        temperature_value,
        "outgoing effective temperature",
    )
    if flux == 0.0 or temperature == 0.0:
        if (
            flux != 0.0
            or temperature != 0.0
            or math.copysign(1.0, flux) != 1.0
            or math.copysign(1.0, temperature) != 1.0
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "zero outgoing flux and temperature must be exact positive-sign "
                "zeros together"
            )
        return 0.0
    try:
        square = temperature * temperature
        reconstructed = STEFAN_BOLTZMANN_W_M2_K4 * square * square
    except OverflowError as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 overflowed binary64"
        ) from error
    if not math.isfinite(reconstructed) or reconstructed <= 0.0:
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 is not finite and positive"
        )
    residual = abs(reconstructed - flux) / max(reconstructed, flux)
    if not math.isfinite(residual) or residual < 0.0:
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 relative residual is invalid"
        )
    if residual > MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL:
        raise KerrReturningRadiationFrameContextVerificationError(
            "sigma*T^4 closure exceeds the non-relaxable relative cap"
        )
    return residual


def _build_d20_certificate(
    outgoing_flux_w_m2: tuple[float, ...],
    outgoing_effective_temperature_k: tuple[float, ...],
) -> ReturningThermalD20FluxCertificateV1:
    if (
        type(outgoing_flux_w_m2) is not tuple
        or type(outgoing_effective_temperature_k) is not tuple
        or not outgoing_flux_w_m2
        or len(outgoing_flux_w_m2) != len(outgoing_effective_temperature_k)
    ):
        raise TypeError("Fout and Teff must be non-empty equal-length exact tuples")
    law = FluxConservingLinearLimbDarkening()
    if type(law) is not FluxConservingLinearLimbDarkening:
        raise KerrReturningRadiationFrameContextVerificationError(
            "D20 law has a non-exact implementation type"
        )
    coefficient = _exact_float(
        object.__getattribute__(law, "coefficient"),
        "D20 coefficient",
    )
    normalization = _exact_float(law.normalization, "D20 normalization")
    hemisphere = 2.0 * normalization * (0.5 + coefficient / 3.0)
    residuals = tuple(
        _sigma_t4_relative_residual(flux, temperature)
        for flux, temperature in zip(
            outgoing_flux_w_m2,
            outgoing_effective_temperature_k,
        )
    )
    values = {
        "coefficient": coefficient,
        "normalization": normalization,
        "hemisphere_flux_normalization": hemisphere,
        "minimum_intensity_multiplier": law.intensity_multiplier(0.0),
        "maximum_intensity_multiplier": law.intensity_multiplier(1.0),
        "sigma_t4_relative_residuals": residuals,
        "maximum_sigma_t4_relative_residual": max(residuals),
        "maximum_allowed_sigma_t4_relative_residual": (
            MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL
        ),
        "one_face": True,
        "implicit_factor_of_two": False,
    }
    provisional = ReturningThermalD20FluxCertificateV1(
        **values,
        _descriptor_json="",
        _descriptor_sha256="",
    )
    # The dataclass cannot call revalidate from __post_init__: construction
    # needs the canonical descriptor that is a function of all prior fields.
    descriptor_json = _canonical_json(_d20_certificate_descriptor(provisional))
    result = ReturningThermalD20FluxCertificateV1(
        **values,
        _descriptor_json=descriptor_json,
        _descriptor_sha256=_sha256_text(descriptor_json),
    )
    result.revalidate()
    return result


def _descriptor_identity(value: Any, label: str) -> tuple[str, str]:
    descriptor_json = object.__getattribute__(value, "_descriptor_json")
    descriptor_sha = object.__getattribute__(value, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} descriptor identity has non-exact types"
        )
    _canonical_embedded_json(descriptor_json, f"{label} descriptor")
    expected_sha = _sha256_text(descriptor_json)
    _digest(descriptor_sha, f"{label} descriptor SHA-256")
    if descriptor_sha.encode("ascii") != expected_sha.encode("ascii"):
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} descriptor SHA-256 is inconsistent"
        )
    return descriptor_json, descriptor_sha


def _surface_identity_descriptors(
    surface: KerrFiniteThicknessMultiSurface,
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be an exact KerrFiniteThicknessMultiSurface")
    metric = object.__getattribute__(surface, "metric")
    calibration = object.__getattribute__(surface, "calibration")
    surface_ids = object.__getattribute__(surface, "surface_ids")
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("surface.metric must be an exact KerrKerrSchildMetric")
    if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "surface.calibration must be an exact "
            "StationaryKerrFiniteThicknessCalibration"
        )
    _require_exact_tree(
        surface_ids,
        FINITE_THICKNESS_SURFACE_IDS,
        "surface.surface_ids",
    )
    rebuilt_metric = KerrKerrSchildMetric(
        mass_m=_exact_positive_float(
            object.__getattribute__(metric, "mass_m"),
            "surface.metric.mass_m",
        ),
        spin_a_m=_exact_float(
            object.__getattribute__(metric, "spin_a_m"),
            "surface.metric.spin_a_m",
        ),
        singularity_guard_m=_exact_positive_float(
            object.__getattribute__(metric, "singularity_guard_m"),
            "surface.metric.singularity_guard_m",
        ),
        source_id=object.__getattribute__(metric, "source_id"),
        time_dependent=object.__getattribute__(metric, "time_dependent"),
    )
    _require_exact_tree(metric, rebuilt_metric, "surface.metric")
    rebuilt_calibration = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=_exact_nonnegative_float(
            object.__getattribute__(calibration, "dimensionless_spin"),
            "surface.calibration.dimensionless_spin",
        ),
        eddington_scaled_mass_accretion_rate=_exact_nonnegative_float(
            object.__getattribute__(
                calibration,
                "eddington_scaled_mass_accretion_rate",
            ),
            "surface.calibration.eddington_scaled_mass_accretion_rate",
        ),
        orientation=object.__getattribute__(calibration, "orientation"),
        outer_radius_over_mass=_exact_positive_float(
            object.__getattribute__(calibration, "outer_radius_over_mass"),
            "surface.calibration.outer_radius_over_mass",
        ),
        thinness_gate_maximum_h_over_rho=_exact_positive_float(
            object.__getattribute__(
                calibration,
                "thinness_gate_maximum_h_over_rho",
            ),
            "surface.calibration.thinness_gate_maximum_h_over_rho",
        ),
    )
    _require_exact_tree(
        calibration,
        rebuilt_calibration,
        "surface.calibration",
    )
    metric_descriptor = {
        "dimensionlessSpin": rebuilt_metric.dimensionless_spin,
        "implementationId": "analytic-kerr-kerr-schild",
        "massM": rebuilt_metric.mass_m,
        "singularityGuardM": rebuilt_metric.singularity_guard_m,
        "sourceId": rebuilt_metric.source_id,
        "spinAM": rebuilt_metric.spin_a_m,
        "timeDependent": rebuilt_metric.time_dependent,
    }
    calibration_descriptor = {
        "dimensionlessSpin": rebuilt_calibration.dimensionless_spin,
        "eddingtonScaledMassAccretionRate": (
            rebuilt_calibration.eddington_scaled_mass_accretion_rate
        ),
        "implementationId": (
            "zhou-2020-taylor-reynolds-stationary-finite-thickness/v1"
        ),
        "iscoRadiusOverMass": rebuilt_calibration.isco_radius_over_mass,
        "orientation": rebuilt_calibration.orientation,
        "outerRadiusOverMass": rebuilt_calibration.outer_radius_over_mass,
        "thinnessGateMaximumHOverRho": (
            rebuilt_calibration.thinness_gate_maximum_h_over_rho
        ),
    }
    metric_json = _canonical_json(metric_descriptor)
    calibration_json = _canonical_json(calibration_descriptor)
    surface_descriptor = {
        "calibrationDescriptorSha256": _sha256_text(calibration_json),
        "exactType": (
            "offline.kerr_finite_thickness_surface."
            "KerrFiniteThicknessMultiSurface"
        ),
        "metricDescriptorSha256": _sha256_text(metric_json),
        "surfaceIds": surface_ids,
    }
    surface_json = _canonical_json(surface_descriptor)
    return (
        (metric_json, _sha256_text(metric_json)),
        (calibration_json, _sha256_text(calibration_json)),
        (surface_json, _sha256_text(surface_json)),
    )


def _disk_identity_descriptor(
    disk: StationaryNovikovThorneDisk,
) -> tuple[str, str]:
    if type(disk) is not StationaryNovikovThorneDisk:
        raise TypeError("profile disk must be an exact StationaryNovikovThorneDisk")
    metric = object.__getattribute__(disk, "metric")
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("profile disk metric must have its exact Kerr type")
    descriptor = {
        "blackHoleMassKg": _exact_positive_float(
            object.__getattribute__(disk, "black_hole_mass_kg"),
            "disk.black_hole_mass_kg",
        ),
        "colourCorrection": _exact_positive_float(
            object.__getattribute__(disk, "colour_correction"),
            "disk.colour_correction",
        ),
        "fluxBoundaryCondition": "zero torque at ISCO",
        "fluxFaceSemantics": "one disk face",
        "massAccretionRateKgS": _exact_nonnegative_float(
            object.__getattribute__(disk, "mass_accretion_rate_kg_s"),
            "disk.mass_accretion_rate_kg_s",
        ),
        "metric": {
            "dimensionlessSpin": metric.dimensionless_spin,
            "massM": object.__getattribute__(metric, "mass_m"),
            "singularityGuardM": object.__getattribute__(
                metric,
                "singularity_guard_m",
            ),
            "sourceId": object.__getattribute__(metric, "source_id"),
            "spinAM": object.__getattribute__(metric, "spin_a_m"),
            "timeDependent": object.__getattribute__(metric, "time_dependent"),
        },
        "orientation": object.__getattribute__(disk, "orientation"),
        "radialScalar": "Page-Thorne 4 pi M^2 F / dot(M)",
        "thermalProvider": "offline.kerr_disk.StationaryNovikovThorneDisk",
    }
    descriptor_json = _canonical_json(descriptor)
    return descriptor_json, _sha256_text(descriptor_json)


def _rebuild_snapshot_identity_descriptors(
    identities: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, tuple[str, str]]:
    """Reconstruct exact identity owners from an untrusted JSON snapshot."""

    try:
        metric_value = identities["metric"]
        calibration_value = identities["calibration"]
        disk_value = identities["disk"]
        disk_metric_value = disk_value["metric"]
        if not all(
            type(value) is dict
            for value in (
                metric_value,
                calibration_value,
                identities["surface"],
                disk_value,
                disk_metric_value,
            )
        ):
            raise TypeError("identity descriptor members must be exact objects")
        metric = KerrKerrSchildMetric(
            mass_m=metric_value["massM"],
            spin_a_m=metric_value["spinAM"],
            singularity_guard_m=metric_value["singularityGuardM"],
            source_id=metric_value["sourceId"],
            time_dependent=metric_value["timeDependent"],
        )
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=calibration_value["dimensionlessSpin"],
            eddington_scaled_mass_accretion_rate=(
                calibration_value["eddingtonScaledMassAccretionRate"]
            ),
            orientation=calibration_value["orientation"],
            outer_radius_over_mass=calibration_value["outerRadiusOverMass"],
            thinness_gate_maximum_h_over_rho=(
                calibration_value["thinnessGateMaximumHOverRho"]
            ),
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        disk_metric = KerrKerrSchildMetric(
            mass_m=disk_metric_value["massM"],
            spin_a_m=disk_metric_value["spinAM"],
            singularity_guard_m=disk_metric_value["singularityGuardM"],
            source_id=disk_metric_value["sourceId"],
            time_dependent=disk_metric_value["timeDependent"],
        )
        disk = StationaryNovikovThorneDisk(
            metric=disk_metric,
            black_hole_mass_kg=disk_value["blackHoleMassKg"],
            mass_accretion_rate_kg_s=disk_value["massAccretionRateKgS"],
            orientation=disk_value["orientation"],
            colour_correction=disk_value["colourCorrection"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot identity descriptor cannot reconstruct exact owners"
        ) from error
    metric_identity, calibration_identity, surface_identity = (
        _surface_identity_descriptors(surface)
    )
    disk_identity = _disk_identity_descriptor(disk)
    return {
        "metric": metric_identity,
        "calibration": calibration_identity,
        "surface": surface_identity,
        "disk": disk_identity,
    }


@dataclass(frozen=True, slots=True)
class ReturningThermalEmissionSnapshotV1:
    """Pickle-safe, live-object-free authenticated emission input."""

    annulus_edges_over_mass: tuple[float, ...]
    outgoing_flux_w_m2: tuple[float, ...]
    outgoing_effective_temperature_k: tuple[float, ...]
    colour_correction: float
    provider_descriptor_sha256: str
    profile_descriptor_sha256: str
    axisymmetric_kernel_descriptor_sha256: str
    source_evidence_descriptor_sha256: str
    source_kernel_descriptor_sha256: str
    underlying_kernel_descriptor_sha256: str
    source_kernel_kind: str
    novikov_thorne_disk_descriptor_sha256: str
    metric_identity_descriptor_json: str
    metric_identity_descriptor_sha256: str
    calibration_identity_descriptor_json: str
    calibration_identity_descriptor_sha256: str
    surface_identity_descriptor_json: str
    surface_identity_descriptor_sha256: str
    disk_identity_descriptor_json: str
    d20_certificate: ReturningThermalD20FluxCertificateV1
    source_closure: tuple[ReturningThermalSourceClosureEntryV1, ...]
    source_closure_manifest_sha256: str
    _descriptor_json: str
    _descriptor_sha256: str

    @property
    def annulus_count(self) -> int:
        return len(self.outgoing_flux_w_m2)

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        self.revalidate()
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        _validate_snapshot(self)


def _snapshot_descriptor(
    snapshot: ReturningThermalEmissionSnapshotV1,
) -> Mapping[str, Any]:
    edges_raw = snapshot.annulus_edges_over_mass
    flux_raw = snapshot.outgoing_flux_w_m2
    temperatures_raw = snapshot.outgoing_effective_temperature_k
    if type(edges_raw) is not tuple or type(flux_raw) is not tuple:
        raise TypeError("snapshot annulus arrays must be exact tuples")
    if type(temperatures_raw) is not tuple or not flux_raw:
        raise TypeError("snapshot temperature array must be a non-empty exact tuple")
    if len(flux_raw) > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot exceeds the fixed authenticated-annulus count limit"
        )
    if len(edges_raw) != len(flux_raw) + 1 or len(flux_raw) != len(temperatures_raw):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot annulus arrays have inconsistent shapes"
        )
    edges = tuple(
        _exact_positive_float(value, f"snapshot edge {index}")
        for index, value in enumerate(edges_raw)
    )
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot annulus edges are not strictly increasing"
        )
    fluxes = tuple(
        _exact_nonnegative_float(value, f"snapshot Fout {index}")
        for index, value in enumerate(flux_raw)
    )
    temperatures = tuple(
        _exact_nonnegative_float(value, f"snapshot Teff {index}")
        for index, value in enumerate(temperatures_raw)
    )
    correction = _exact_positive_float(
        snapshot.colour_correction,
        "snapshot colour correction",
    )
    if correction < 1.0:
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot colour correction must be at least one"
        )
    hash_names = (
        "provider_descriptor_sha256",
        "profile_descriptor_sha256",
        "axisymmetric_kernel_descriptor_sha256",
        "source_evidence_descriptor_sha256",
        "source_kernel_descriptor_sha256",
        "underlying_kernel_descriptor_sha256",
        "novikov_thorne_disk_descriptor_sha256",
        "metric_identity_descriptor_sha256",
        "calibration_identity_descriptor_sha256",
        "surface_identity_descriptor_sha256",
        "source_closure_manifest_sha256",
    )
    for name in hash_names:
        _digest(object.__getattribute__(snapshot, name), f"snapshot.{name}")
    source_kind = snapshot.source_kernel_kind
    if type(source_kind) is not str or source_kind not in (
        "KerrForwardReturningRadiationKernel",
        "KerrForwardReturningRadiationKernelProjection",
        "KerrCachedReturningRadiationKernelExecution",
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot source kernel kind is not supported"
        )
    if source_kind in (
        "KerrForwardReturningRadiationKernel",
        "KerrForwardReturningRadiationKernelProjection",
    ) and (
        snapshot.source_evidence_descriptor_sha256.encode("ascii")
        != snapshot.source_kernel_descriptor_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "direct source evidence does not equal its effective kernel descriptor"
        )
    if source_kind in (
        "KerrForwardReturningRadiationKernel",
        "KerrCachedReturningRadiationKernelExecution",
    ) and (
        snapshot.source_kernel_descriptor_sha256.encode("ascii")
        != snapshot.underlying_kernel_descriptor_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "full source effective and underlying kernel descriptors differ"
        )
    identity_values: dict[str, Mapping[str, Any]] = {}
    for label, json_name, sha_name in (
        (
            "metric",
            "metric_identity_descriptor_json",
            "metric_identity_descriptor_sha256",
        ),
        (
            "calibration",
            "calibration_identity_descriptor_json",
            "calibration_identity_descriptor_sha256",
        ),
        (
            "surface",
            "surface_identity_descriptor_json",
            "surface_identity_descriptor_sha256",
        ),
        (
            "disk",
            "disk_identity_descriptor_json",
            "novikov_thorne_disk_descriptor_sha256",
        ),
    ):
        canonical, parsed = _canonical_embedded_json(
            object.__getattribute__(snapshot, json_name),
            f"snapshot {label} identity",
        )
        expected_sha = _sha256_text(canonical)
        actual_sha = object.__getattribute__(snapshot, sha_name)
        if expected_sha.encode("ascii") != actual_sha.encode("ascii"):
            raise KerrReturningRadiationFrameContextVerificationError(
                f"snapshot {label} identity SHA-256 is inconsistent"
            )
        identity_values[label] = parsed
    rebuilt_identities = _rebuild_snapshot_identity_descriptors(identity_values)
    for label, json_name, sha_name in (
        (
            "metric",
            "metric_identity_descriptor_json",
            "metric_identity_descriptor_sha256",
        ),
        (
            "calibration",
            "calibration_identity_descriptor_json",
            "calibration_identity_descriptor_sha256",
        ),
        (
            "surface",
            "surface_identity_descriptor_json",
            "surface_identity_descriptor_sha256",
        ),
        (
            "disk",
            "disk_identity_descriptor_json",
            "novikov_thorne_disk_descriptor_sha256",
        ),
    ):
        expected_json, expected_sha = rebuilt_identities[label]
        if (
            object.__getattribute__(snapshot, json_name).encode("utf-8")
            != expected_json.encode("utf-8")
            or object.__getattribute__(snapshot, sha_name).encode("ascii")
            != expected_sha.encode("ascii")
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                f"snapshot {label} identity differs from exact reconstruction"
            )
    certificate = snapshot.d20_certificate
    if type(certificate) is not ReturningThermalD20FluxCertificateV1:
        raise TypeError("snapshot D20 certificate must have its exact type")
    certificate.revalidate()
    if len(certificate.sigma_t4_relative_residuals) != len(fluxes):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot D20 residual count differs from its annuli"
        )
    replayed_certificate = _build_d20_certificate(fluxes, temperatures)
    _require_exact_tree(
        certificate,
        replayed_certificate,
        "snapshot.d20_certificate",
    )
    closure = snapshot.source_closure
    _validate_source_closure_resources(closure, "snapshot source closure")
    expected_paths = tuple(sorted((*_SOURCE_CLOSURE_PATHS, _NUMERIC_BACKEND_LOGICAL_PATH)))
    actual_paths = tuple(entry.logical_path for entry in closure)
    if actual_paths != expected_paths:
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot source closure paths are not the fixed transitive set"
        )
    expected_manifest_sha = _source_closure_manifest_sha256(closure)
    if (
        expected_manifest_sha.encode("ascii")
        != snapshot.source_closure_manifest_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot source-closure manifest SHA-256 is inconsistent"
        )
    calibration_identity = identity_values["calibration"]
    metric_identity = identity_values["metric"]
    surface_identity = identity_values["surface"]
    disk_identity = identity_values["disk"]
    try:
        if edges[0].hex() != float(
            calibration_identity["iscoRadiusOverMass"]
        ).hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot inner edge differs from calibrated ISCO"
            )
        if edges[-1].hex() != float(
            calibration_identity["outerRadiusOverMass"]
        ).hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot outer edge differs from calibrated R_out"
            )
        if (
            surface_identity["metricDescriptorSha256"]
            != snapshot.metric_identity_descriptor_sha256
            or surface_identity["calibrationDescriptorSha256"]
            != snapshot.calibration_identity_descriptor_sha256
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot surface identity does not bind metric/calibration"
            )
        disk_metric = disk_identity["metric"]
        if float(disk_metric["massM"]).hex() != float(metric_identity["massM"]).hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot disk and surface metric masses differ"
            )
        if float(disk_metric["spinAM"]).hex() != float(metric_identity["spinAM"]).hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot disk and surface signed spins differ"
            )
        if float(disk_metric["singularityGuardM"]).hex() != float(
            metric_identity["singularityGuardM"]
        ).hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot disk and surface singularity guards differ"
            )
        if disk_identity["orientation"] != calibration_identity["orientation"]:
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot disk and calibration orientations differ"
            )
        if float(disk_identity["colourCorrection"]).hex() != correction.hex():
            raise KerrReturningRadiationFrameContextVerificationError(
                "snapshot disk and provider colour corrections differ"
            )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, KerrReturningRadiationFrameContextVerificationError):
            raise
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot identity descriptor schema is incomplete"
        ) from error
    return {
        "annuli": {
            "edgesOverMass": edges,
            "outgoingEffectiveTemperatureK": temperatures,
            "outgoingFluxWM2": fluxes,
            "valueModel": "piecewise constant authenticated thermal profile",
        },
        "binding": {
            "axisymmetricKernelDescriptorSha256": (
                snapshot.axisymmetric_kernel_descriptor_sha256
            ),
            "calibrationIdentity": identity_values["calibration"],
            "diskIdentity": identity_values["disk"],
            "metricIdentity": identity_values["metric"],
            "novikovThorneDiskDescriptorSha256": (
                snapshot.novikov_thorne_disk_descriptor_sha256
            ),
            "profileDescriptorSha256": snapshot.profile_descriptor_sha256,
            "providerDescriptorSha256": snapshot.provider_descriptor_sha256,
            "sourceKernelDescriptorSha256": (
                snapshot.source_kernel_descriptor_sha256
            ),
            "sourceEvidenceDescriptorSha256": (
                snapshot.source_evidence_descriptor_sha256
            ),
            "sourceKernelKind": source_kind,
            "surfaceIdentity": identity_values["surface"],
            "underlyingKernelDescriptorSha256": (
                snapshot.underlying_kernel_descriptor_sha256
            ),
        },
        "emission": {
            "colourCorrection": correction,
            "d20FluxCertificate": certificate.model_descriptor(),
            "frequencyFrame": "local-comoving emitter frame",
            "oneFace": True,
        },
        "implementationId": SNAPSHOT_IMPLEMENTATION_ID,
        "scientificBoundary": {
            "hasIndependentPhysicsOracle": False,
            "includesAtmosphere": False,
            "includesPolarization": False,
            "includesReturningRadiationStressWorkFS": False,
            "includesScattering": False,
            "includesSpectralRedistribution": False,
            "isArtifactLoader": False,
            "isCompleteKerrbb": False,
            "isSameCodeAuthenticatedInput": True,
        },
        "sourceClosure": {
            "entries": tuple(entry.descriptor() for entry in closure),
            "manifestSha256": snapshot.source_closure_manifest_sha256,
            "recheckedBeforeAndAfterAuthentication": True,
            "runtimeBackendIncluded": True,
        },
        "units": {
            "flux": "W m^-2 local-comoving one-face proper area",
            "temperature": "K",
        },
    }


def _validate_snapshot(snapshot: ReturningThermalEmissionSnapshotV1) -> None:
    if type(snapshot) is not ReturningThermalEmissionSnapshotV1:
        raise TypeError("snapshot must have its exact public type")
    descriptor = _snapshot_descriptor(snapshot)
    expected_json = _canonical_json(descriptor)
    supplied_json = object.__getattribute__(snapshot, "_descriptor_json")
    supplied_sha = object.__getattribute__(snapshot, "_descriptor_sha256")
    if type(supplied_json) is not str or type(supplied_sha) is not str:
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot descriptor identity has non-exact types"
        )
    if (
        supplied_json.encode("utf-8") != expected_json.encode("utf-8")
        or supplied_sha.encode("ascii")
        != _sha256_text(expected_json).encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "snapshot differs from its canonical descriptor tree"
        )


@dataclass(frozen=True, slots=True)
class _AuthenticatedLiveInputs:
    surface: KerrFiniteThicknessMultiSurface
    kernel_surface: KerrFiniteThicknessMultiSurface
    provider: CertifiedReturningRadiationThermalSpectrumProvider
    profile: AxisymmetricReturningRadiationThermalProfile
    source_kernel: (
        KerrForwardReturningRadiationKernel
        | KerrForwardReturningRadiationKernelProjection
        | KerrCachedReturningRadiationKernelExecution
    )
    effective_kernel: (
        KerrForwardReturningRadiationKernel
        | KerrForwardReturningRadiationKernelProjection
    )
    underlying_kernel: KerrForwardReturningRadiationKernel
    axisymmetric_kernel: AxisymmetricReturningRadiationKernel
    disk: StationaryNovikovThorneDisk


def _resolve_certified_source_graph(
    source: Any,
) -> tuple[
    KerrForwardReturningRadiationKernel
    | KerrForwardReturningRadiationKernelProjection,
    KerrForwardReturningRadiationKernel,
    str,
]:
    """Resolve exact source ownership without I/O, replay, or descriptors."""

    if type(source) is KerrForwardReturningRadiationKernel:
        return source, source, "KerrForwardReturningRadiationKernel"
    if type(source) is KerrForwardReturningRadiationKernelProjection:
        underlying = object.__getattribute__(source, "_source")
        if type(underlying) is not KerrForwardReturningRadiationKernel:
            raise KerrReturningRadiationFrameContextVerificationError(
                "kernel projection does not own an exact full forward source"
            )
        return (
            source,
            underlying,
            "KerrForwardReturningRadiationKernelProjection",
        )
    if type(source) is KerrCachedReturningRadiationKernelExecution:
        formulation = object.__getattribute__(source, "formulation")
        kernel = object.__getattribute__(source, "kernel")
        if (
            type(formulation) is not str
            or formulation.encode("utf-8") != FORWARD.encode("utf-8")
            or type(kernel) is not KerrForwardReturningRadiationKernel
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "cached thermal source is not an exact forward execution"
            )
        return (
            kernel,
            kernel,
            "KerrCachedReturningRadiationKernelExecution",
        )
    raise KerrReturningRadiationFrameContextVerificationError(
        "profile source has no supported exact certified forward evidence"
    )


def _preflight_annulus_count_from_edges(value: Any, label: str) -> int:
    if type(value) is not tuple or len(value) < 2:
        raise TypeError(f"{label} must be an exact edge tuple with at least two values")
    count = len(value) - 1
    if count > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} exceeds the fixed authenticated-annulus count limit"
        )
    return count


def _preflight_annular_vector(value: Any, count: int, label: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    if len(value) != count:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} length differs from its preflight annulus count"
        )


def _preflight_square_matrix(value: Any, count: int, label: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple of exact tuples")
    if len(value) != count:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"{label} row count differs from its preflight annulus count"
        )
    for row_index, row in enumerate(value):
        if type(row) is not tuple or len(row) != count:
            raise KerrReturningRadiationFrameContextVerificationError(
                f"{label}[{row_index}] has an invalid preflight column count"
            )


def _preflight_authentication_resources(
    surface: KerrFiniteThicknessMultiSurface,
    provider: CertifiedReturningRadiationThermalSpectrumProvider,
) -> None:
    """Bound attacker-controlled graph shapes before source I/O or replay."""

    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be an exact KerrFiniteThicknessMultiSurface")
    if type(provider) is not CertifiedReturningRadiationThermalSpectrumProvider:
        raise TypeError(
            "provider must be an exact "
            "CertifiedReturningRadiationThermalSpectrumProvider"
        )
    try:
        provider_count = _preflight_annulus_count_from_edges(
            object.__getattribute__(provider, "annulus_edges_over_mass"),
            "provider annulus edges",
        )
        _preflight_annular_vector(
            object.__getattribute__(
                provider,
                "outgoing_effective_temperature_k",
            ),
            provider_count,
            "provider outgoing temperatures",
        )
        profile = object.__getattribute__(provider, "_profile")
        if type(profile) is not AxisymmetricReturningRadiationThermalProfile:
            raise TypeError("provider profile must have its exact certified type")
        profile_count = _preflight_annulus_count_from_edges(
            object.__getattribute__(profile, "annulus_edges_over_mass"),
            "profile annulus edges",
        )
        _preflight_annular_vector(
            object.__getattribute__(profile, "outgoing_flux_w_m2"),
            profile_count,
            "profile outgoing fluxes",
        )
        _preflight_annular_vector(
            object.__getattribute__(
                profile,
                "outgoing_effective_temperature_k",
            ),
            profile_count,
            "profile outgoing temperatures",
        )
        if provider_count != profile_count:
            raise KerrReturningRadiationFrameContextVerificationError(
                "provider/profile preflight annulus counts differ"
            )
        source = object.__getattribute__(profile, "_certified_source")
        effective, underlying, _source_kind = _resolve_certified_source_graph(
            source
        )
        source_count = _preflight_annulus_count_from_edges(
            object.__getattribute__(effective, "annulus_edges_over_mass"),
            "source-kernel annulus edges",
        )
        if source_count != profile_count:
            raise KerrReturningRadiationFrameContextVerificationError(
                "profile/source-kernel preflight annulus counts differ"
            )
        for matrix_name in (
            "upper_receiver_upper_emitter_coefficients",
            "upper_receiver_lower_emitter_coefficients",
            "lower_receiver_upper_emitter_coefficients",
            "lower_receiver_lower_emitter_coefficients",
        ):
            _preflight_square_matrix(
                object.__getattribute__(effective, matrix_name),
                source_count,
                f"source-kernel {matrix_name}",
            )
        # Cached execution wrappers carry matrices on their rebuilt kernel,
        # never on the execution object itself.
        if type(underlying) is not KerrForwardReturningRadiationKernel:
            raise TypeError("underlying source must have its exact full-kernel type")
        underlying_count = _preflight_annulus_count_from_edges(
            object.__getattribute__(underlying, "annulus_edges_over_mass"),
            "underlying-kernel annulus edges",
        )
        for matrix_name in (
            "upper_receiver_upper_emitter_coefficients",
            "upper_receiver_lower_emitter_coefficients",
            "lower_receiver_upper_emitter_coefficients",
            "lower_receiver_lower_emitter_coefficients",
        ):
            _preflight_square_matrix(
                object.__getattribute__(underlying, matrix_name),
                underlying_count,
                f"underlying-kernel {matrix_name}",
            )
        axisymmetric = object.__getattribute__(profile, "_kernel")
        if type(axisymmetric) is not AxisymmetricReturningRadiationKernel:
            raise TypeError("profile axisymmetric kernel must have its exact type")
        axis_radii = object.__getattribute__(
            axisymmetric,
            "annulus_radii_over_mass",
        )
        if type(axis_radii) is not tuple or not axis_radii:
            raise TypeError("axisymmetric kernel radii must be a non-empty exact tuple")
        axis_count = len(axis_radii)
        if axis_count > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
            raise KerrReturningRadiationFrameContextVerificationError(
                "axisymmetric kernel exceeds the fixed annulus-count limit"
            )
        if axis_count != profile_count:
            raise KerrReturningRadiationFrameContextVerificationError(
                "profile/axisymmetric-kernel preflight annulus counts differ"
            )
        _preflight_square_matrix(
            object.__getattribute__(
                axisymmetric,
                "receiver_emitter_coefficients",
            ),
            axis_count,
            "axisymmetric-kernel coefficients",
        )
    except AttributeError as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            "authentication graph is missing a required preflight field"
        ) from error


def _extract_authenticated_live_inputs(
    surface: KerrFiniteThicknessMultiSurface,
    provider: CertifiedReturningRadiationThermalSpectrumProvider,
) -> _AuthenticatedLiveInputs:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be an exact KerrFiniteThicknessMultiSurface")
    if type(provider) is not CertifiedReturningRadiationThermalSpectrumProvider:
        raise TypeError(
            "provider must be an exact "
            "CertifiedReturningRadiationThermalSpectrumProvider"
        )
    try:
        verify_certified_returning_radiation_thermal_spectrum_provider(provider)
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"thermal provider could not be replay-authenticated: {error}"
        ) from error
    profile = object.__getattribute__(provider, "_profile")
    if type(profile) is not AxisymmetricReturningRadiationThermalProfile:
        raise TypeError("provider profile must have its exact certified type")
    if (
        object.__getattribute__(profile, "geometry_provenance_certified") is not True
        or object.__getattribute__(profile, "provenance_classification")
        not in CERTIFIED_KERR_FORWARD_CLASSIFICATIONS
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "provider profile is not replay-certified Kerr geometry"
        )
    try:
        checked_axisymmetric, checked_disk, checked_policy = (
            _validated_profile_live_owners(profile)
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationFrameContextVerificationError(
            f"provider profile live scientific owners are not canonical: {error}"
        ) from error
    declared_axis_sha = object.__getattribute__(
        profile,
        "kernel_descriptor_sha256",
    )
    declared_policy_sha = object.__getattribute__(
        profile,
        "fixed_point_policy_descriptor_sha256",
    )
    _digest(declared_axis_sha, "profile axisymmetric-kernel descriptor")
    _digest(declared_policy_sha, "profile fixed-point policy descriptor")
    if (
        checked_axisymmetric.canonical_descriptor_sha256.encode("ascii")
        != declared_axis_sha.encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "live axisymmetric kernel differs from the source-certified reduction"
        )
    if (
        checked_policy.canonical_descriptor_sha256.encode("ascii")
        != declared_policy_sha.encode("ascii")
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "live fixed-point policy differs from the source-certified profile"
        )
    source = object.__getattribute__(profile, "_certified_source")
    effective, underlying, source_kind = _resolve_certified_source_graph(source)
    original_surface = object.__getattribute__(underlying, "surface")
    if type(original_surface) is not KerrFiniteThicknessMultiSurface:
        raise KerrReturningRadiationFrameContextVerificationError(
            "underlying full kernel surface has a non-exact type"
        )
    # CLI/worker boundaries may reconstruct the geometry.  Object identity is
    # therefore not a physical requirement here, but the exact dataclass tree
    # is: every binary64 field, orientation string, and stable surface id must
    # agree without invoking overloaded equality.
    _require_exact_tree(
        surface,
        original_surface,
        "render_surface/original_kernel_surface",
    )
    axisymmetric = object.__getattribute__(profile, "_kernel")
    disk = object.__getattribute__(profile, "_disk")
    if type(axisymmetric) is not AxisymmetricReturningRadiationKernel:
        raise TypeError("profile axisymmetric kernel must have its exact type")
    if type(disk) is not StationaryNovikovThorneDisk:
        raise TypeError("profile disk must have its exact type")
    _require_exact_tree(
        axisymmetric,
        checked_axisymmetric,
        "profile.axisymmetric_kernel/canonical_live_owner",
    )
    _require_exact_tree(
        disk,
        checked_disk,
        "profile.disk/canonical_live_owner",
    )
    _surface_identity_descriptors(surface)
    metric = object.__getattribute__(surface, "metric")
    calibration = object.__getattribute__(surface, "calibration")
    disk_metric = object.__getattribute__(disk, "metric")
    for actual, expected, label in (
        (disk_metric.mass_m, metric.mass_m, "metric mass"),
        (disk_metric.spin_a_m, metric.spin_a_m, "signed Kerr spin"),
        (
            disk_metric.singularity_guard_m,
            metric.singularity_guard_m,
            "metric singularity guard",
        ),
        (
            disk.dimensionless_spin_magnitude,
            calibration.dimensionless_spin,
            "calibration spin",
        ),
    ):
        if not _same_float(actual, expected):
            raise KerrReturningRadiationFrameContextVerificationError(
                f"render surface and thermal disk {label} differ"
            )
    if disk.orientation.encode("utf-8") != calibration.orientation.encode("utf-8"):
        raise KerrReturningRadiationFrameContextVerificationError(
            "render calibration and thermal disk orientations differ"
        )
    edges = object.__getattribute__(provider, "annulus_edges_over_mass")
    profile_edges = object.__getattribute__(profile, "annulus_edges_over_mass")
    source_edges = object.__getattribute__(effective, "annulus_edges_over_mass")
    _require_exact_tree(edges, profile_edges, "provider/profile annulus edges")
    _require_exact_tree(edges, source_edges, "provider/source annulus edges")
    if (
        not _same_float(edges[0], calibration.isco_radius_over_mass)
        or not _same_float(edges[-1], calibration.outer_radius_over_mass)
    ):
        raise KerrReturningRadiationFrameContextVerificationError(
            "authenticated annulus domain does not exactly cover ISCO to R_out"
        )
    temperatures = object.__getattribute__(
        provider,
        "outgoing_effective_temperature_k",
    )
    _require_exact_tree(
        temperatures,
        object.__getattribute__(profile, "outgoing_effective_temperature_k"),
        "provider/profile outgoing temperatures",
    )
    correction = object.__getattribute__(provider, "colour_correction")
    if not _same_float(correction, object.__getattribute__(disk, "colour_correction")):
        raise KerrReturningRadiationFrameContextVerificationError(
            "provider and disk colour corrections differ"
        )
    if source_kind not in (
        "KerrForwardReturningRadiationKernel",
        "KerrForwardReturningRadiationKernelProjection",
        "KerrCachedReturningRadiationKernelExecution",
    ):
        raise AssertionError("unreachable source kind")
    return _AuthenticatedLiveInputs(
        surface,
        original_surface,
        provider,
        profile,
        source,
        effective,
        underlying,
        axisymmetric,
        disk,
    )


def _build_snapshot(
    live: _AuthenticatedLiveInputs,
    closure: tuple[ReturningThermalSourceClosureEntryV1, ...],
) -> ReturningThermalEmissionSnapshotV1:
    provider = live.provider
    profile = live.profile
    source = live.source_kernel
    effective = live.effective_kernel
    underlying = live.underlying_kernel
    axisymmetric = live.axisymmetric_kernel
    disk = live.disk
    _provider_json, provider_sha = _descriptor_identity(provider, "provider")
    _profile_json, profile_sha = _descriptor_identity(profile, "profile")
    _effective_json, source_sha = _descriptor_identity(
        effective,
        "effective source kernel",
    )
    if type(source) is KerrCachedReturningRadiationKernelExecution:
        source_evidence_sha = object.__getattribute__(
            profile,
            "source_authentication_descriptor_sha256",
        )
        _digest(
            source_evidence_sha,
            "cached source authentication descriptor SHA-256",
        )
    else:
        _source_json, source_evidence_sha = _descriptor_identity(
            source,
            "source evidence",
        )
    _underlying_json, underlying_sha = _descriptor_identity(
        underlying,
        "underlying kernel",
    )
    _axis_json, axis_sha = _descriptor_identity(
        axisymmetric,
        "axisymmetric kernel",
    )
    metric_identity, calibration_identity, surface_identity = (
        _surface_identity_descriptors(live.surface)
    )
    disk_json, disk_sha = _disk_identity_descriptor(disk)
    declared_disk_sha = object.__getattribute__(
        profile,
        "novikov_thorne_disk_descriptor_sha256",
    )
    _digest(declared_disk_sha, "profile Novikov-Thorne disk descriptor")
    if disk_sha.encode("ascii") != declared_disk_sha.encode("ascii"):
        raise KerrReturningRadiationFrameContextVerificationError(
            "profile disk descriptor differs from its live disk identity"
        )
    if axis_sha.encode("ascii") != object.__getattribute__(
        profile,
        "kernel_descriptor_sha256",
    ).encode("ascii"):
        raise KerrReturningRadiationFrameContextVerificationError(
            "profile axisymmetric kernel descriptor binding is stale"
        )
    if profile_sha.encode("ascii") != object.__getattribute__(
        provider,
        "source_profile_descriptor_sha256",
    ).encode("ascii"):
        raise KerrReturningRadiationFrameContextVerificationError(
            "provider profile descriptor binding is stale"
        )
    edges = object.__getattribute__(provider, "annulus_edges_over_mass")
    temperatures = object.__getattribute__(
        provider,
        "outgoing_effective_temperature_k",
    )
    fluxes = object.__getattribute__(profile, "outgoing_flux_w_m2")
    certificate = _build_d20_certificate(fluxes, temperatures)
    _effective_check, _underlying_check, source_kind = (
        _resolve_certified_source_graph(source)
    )
    values = {
        "annulus_edges_over_mass": tuple(edges),
        "outgoing_flux_w_m2": tuple(fluxes),
        "outgoing_effective_temperature_k": tuple(temperatures),
        "colour_correction": object.__getattribute__(provider, "colour_correction"),
        "provider_descriptor_sha256": provider_sha,
        "profile_descriptor_sha256": profile_sha,
        "axisymmetric_kernel_descriptor_sha256": axis_sha,
        "source_evidence_descriptor_sha256": source_evidence_sha,
        "source_kernel_descriptor_sha256": source_sha,
        "underlying_kernel_descriptor_sha256": underlying_sha,
        "source_kernel_kind": source_kind,
        "novikov_thorne_disk_descriptor_sha256": disk_sha,
        "metric_identity_descriptor_json": metric_identity[0],
        "metric_identity_descriptor_sha256": metric_identity[1],
        "calibration_identity_descriptor_json": calibration_identity[0],
        "calibration_identity_descriptor_sha256": calibration_identity[1],
        "surface_identity_descriptor_json": surface_identity[0],
        "surface_identity_descriptor_sha256": surface_identity[1],
        "disk_identity_descriptor_json": disk_json,
        "d20_certificate": certificate,
        "source_closure": closure,
        "source_closure_manifest_sha256": (
            _source_closure_manifest_sha256(closure)
        ),
    }
    provisional = ReturningThermalEmissionSnapshotV1(
        **values,
        _descriptor_json="",
        _descriptor_sha256="",
    )
    descriptor_json = _canonical_json(_snapshot_descriptor(provisional))
    result = ReturningThermalEmissionSnapshotV1(
        **values,
        _descriptor_json=descriptor_json,
        _descriptor_sha256=_sha256_text(descriptor_json),
    )
    result.revalidate()
    return result


@dataclass(frozen=True, slots=True)
class _AuthorityRecord:
    live: _AuthenticatedLiveInputs
    snapshot: ReturningThermalEmissionSnapshotV1
    source_closure: tuple[ReturningThermalSourceClosureEntryV1, ...]
    provider_live_identity_sha256: str
    render_surface_live_identity_sha256: str
    emission_edges_over_mass: tuple[float, ...]
    emission_effective_temperature_k: tuple[float, ...]
    emission_colour_correction: float


def _make_authority_api():
    registry: weakref.WeakKeyDictionary[Any, _AuthorityRecord] = (
        weakref.WeakKeyDictionary()
    )
    lock = threading.RLock()
    planck_kernel = colour_corrected_planck_specific_intensity_nu

    class ValidatedReturningThermalAuthority:
        """Opaque process-local ownership proof for one live emission graph."""

        __slots__ = ("__weakref__",)

        def __new__(cls):
            raise TypeError(
                "ValidatedReturningThermalAuthority is created only by "
                "authenticate_returning_thermal_emission"
            )

        def __init_subclass__(cls, **kwargs):
            del kwargs
            raise TypeError("ValidatedReturningThermalAuthority cannot be subclassed")

        @property
        def snapshot(self) -> ReturningThermalEmissionSnapshotV1:
            with lock:
                try:
                    return registry[self].snapshot
                except (KeyError, TypeError) as error:
                    raise KerrReturningRadiationFrameContextVerificationError(
                        "authority is not registered in this process"
                    ) from error

        def require_live(self) -> ReturningThermalEmissionSnapshotV1:
            """Recheck the bound live graph without replaying the ray kernel."""

            with lock:
                try:
                    record = registry[self]
                except (KeyError, TypeError) as error:
                    raise KerrReturningRadiationFrameContextVerificationError(
                        "authority is not registered in this process"
                    ) from error
            return require_authority(
                self,
                record.live.surface,
                record.live.provider,
            )

        def emitted_specific_intensity_nu_batch(
            self,
            radius_over_mass: float,
            emitted_frequencies_hz: tuple[float, ...],
        ) -> tuple[float, ...]:
            """Consume the private frozen table without reading live physics.

            Call :meth:`require_live` at explicit frame boundaries.  Per-ray
            batches only authenticate the closure-private primitive table,
            resolve one annulus, and evaluate the shared Planck kernel.
            """

            radius = _exact_float(radius_over_mass, "radius_over_mass")
            if type(emitted_frequencies_hz) is not tuple or not emitted_frequencies_hz:
                raise TypeError(
                    "emitted_frequencies_hz must be a non-empty exact tuple"
                )
            if len(emitted_frequencies_hz) > 4096:
                raise ValueError("frequency batch exceeds the fixed 4096-bin limit")
            frequencies = tuple(
                _exact_positive_float(value, f"emitted frequency {index}")
                for index, value in enumerate(emitted_frequencies_hz)
            )
            with lock:
                try:
                    record = registry[self]
                except (KeyError, TypeError) as error:
                    raise KerrReturningRadiationFrameContextVerificationError(
                        "authority is not registered in this process"
                    ) from error
            edges = record.emission_edges_over_mass
            temperatures = record.emission_effective_temperature_k
            correction = record.emission_colour_correction
            if radius < edges[0]:
                raise ValueError("radius lies inside the authenticated profile domain")
            if radius > edges[-1]:
                raise ValueError("radius lies outside the authenticated profile domain")
            annulus_index = bisect_right(edges, radius) - 1
            if annulus_index == len(temperatures):
                annulus_index -= 1
            effective_temperature = temperatures[annulus_index]
            values = tuple(
                planck_kernel(
                    effective_temperature,
                    correction,
                    frequency,
                )
                for frequency in frequencies
            )
            checked = tuple(
                _exact_nonnegative_float(value, f"specific intensity {index}")
                for index, value in enumerate(values)
            )
            with lock:
                try:
                    after_record = registry[self]
                except (KeyError, TypeError) as error:
                    raise KerrReturningRadiationFrameContextVerificationError(
                        "authority disappeared during batch evaluation"
                    ) from error
            if after_record is not record:
                raise KerrReturningRadiationFrameContextVerificationError(
                    "authority registry binding changed during batch evaluation"
                )
            return checked

        def __copy__(self):
            raise TypeError("returning-thermal authority cannot be copied")

        def __deepcopy__(self, memo):
            del memo
            raise TypeError("returning-thermal authority cannot be deep-copied")

        def __reduce__(self):
            raise TypeError("returning-thermal authority cannot be pickled")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("returning-thermal authority cannot be pickled")

    ValidatedReturningThermalAuthority.__name__ = (
        "ValidatedReturningThermalAuthority"
    )
    ValidatedReturningThermalAuthority.__qualname__ = (
        "ValidatedReturningThermalAuthority"
    )
    ValidatedReturningThermalAuthority.__module__ = __name__

    def _stable_current_closure(
        expected: tuple[ReturningThermalSourceClosureEntryV1, ...],
        phase: str,
    ) -> tuple[ReturningThermalSourceClosureEntryV1, ...]:
        current = _source_closure_manifest()
        try:
            _require_exact_tree(current, expected, phase)
        except KerrReturningRadiationFrameContextVerificationError as error:
            raise KerrReturningRadiationFrameContextVerificationError(
                "authenticated source/runtime closure has changed"
            ) from error
        return current

    def authenticate_returning_thermal_emission(
        surface: KerrFiniteThicknessMultiSurface,
        provider: CertifiedReturningRadiationThermalSpectrumProvider,
    ) -> Any:
        """Replay once, freeze a snapshot, and register a local authority."""

        _preflight_authentication_resources(surface, provider)
        before_closure = _source_closure_manifest()
        live = _extract_authenticated_live_inputs(surface, provider)
        before_provider_identity = _live_identity_sha256(provider, "provider")
        before_surface_identity = _live_identity_sha256(
            surface,
            "render_surface",
        )
        snapshot = _build_snapshot(live, before_closure)
        after_closure = _source_closure_manifest()
        after_provider_identity = _live_identity_sha256(provider, "provider")
        after_surface_identity = _live_identity_sha256(
            surface,
            "render_surface",
        )
        try:
            _require_exact_tree(
                after_closure,
                before_closure,
                "source_closure.after_authentication",
            )
        except KerrReturningRadiationFrameContextVerificationError as error:
            raise KerrReturningRadiationFrameContextVerificationError(
                "source/runtime closure changed while authentication was running"
            ) from error
        if (
            before_provider_identity.encode("ascii")
            != after_provider_identity.encode("ascii")
            or before_surface_identity.encode("ascii")
            != after_surface_identity.encode("ascii")
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "live provider or render surface changed while authentication "
                "was running"
            )
        authority = object.__new__(ValidatedReturningThermalAuthority)
        record = _AuthorityRecord(
            live,
            snapshot,
            before_closure,
            before_provider_identity,
            before_surface_identity,
            tuple(snapshot.annulus_edges_over_mass),
            tuple(snapshot.outgoing_effective_temperature_k),
            snapshot.colour_correction,
        )
        with lock:
            registry[authority] = record
        return authority

    def require_authority(
        authority: Any,
        surface: KerrFiniteThicknessMultiSurface,
        provider: CertifiedReturningRadiationThermalSpectrumProvider,
    ) -> ReturningThermalEmissionSnapshotV1:
        """Cheaply rebind live objects and hashes without replaying any ray."""

        if type(authority) is not ValidatedReturningThermalAuthority:
            raise TypeError("authority must have its exact process-local type")
        if type(surface) is not KerrFiniteThicknessMultiSurface:
            raise TypeError("surface must have its exact finite-thickness type")
        if type(provider) is not CertifiedReturningRadiationThermalSpectrumProvider:
            raise TypeError("provider must have its exact certified spectrum type")
        with lock:
            try:
                record = registry[authority]
            except (KeyError, TypeError) as error:
                raise KerrReturningRadiationFrameContextVerificationError(
                    "authority is not registered in this process"
                ) from error
        if surface is not record.live.surface or provider is not record.live.provider:
            raise KerrReturningRadiationFrameContextVerificationError(
                "authority belongs to different live surface/provider objects"
            )
        _stable_current_closure(
            record.source_closure,
            "source_closure.before_require",
        )
        if object.__getattribute__(provider, "_profile") is not record.live.profile:
            raise KerrReturningRadiationFrameContextVerificationError(
                "provider live profile identity changed after authentication"
            )
        if object.__getattribute__(record.live.profile, "_kernel") is not (
            record.live.axisymmetric_kernel
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "profile live axisymmetric-kernel identity changed after "
                "authentication"
            )
        if object.__getattribute__(record.live.profile, "_disk") is not (
            record.live.disk
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "profile live disk identity changed after authentication"
            )
        if object.__getattribute__(record.live.profile, "_certified_source") is not (
            record.live.source_kernel
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "profile live source-kernel identity changed after authentication"
            )
        if type(record.live.source_kernel) is KerrForwardReturningRadiationKernelProjection:
            if object.__getattribute__(record.live.source_kernel, "_source") is not (
                record.live.underlying_kernel
            ):
                raise KerrReturningRadiationFrameContextVerificationError(
                    "projection underlying-kernel identity changed"
                )
            if record.live.effective_kernel is not record.live.source_kernel:
                raise KerrReturningRadiationFrameContextVerificationError(
                    "projection effective-kernel identity changed"
                )
        elif type(record.live.source_kernel) is (
            KerrCachedReturningRadiationKernelExecution
        ):
            if object.__getattribute__(record.live.source_kernel, "kernel") is not (
                record.live.underlying_kernel
            ):
                raise KerrReturningRadiationFrameContextVerificationError(
                    "cached execution underlying-kernel identity changed"
                )
            if record.live.effective_kernel is not record.live.underlying_kernel:
                raise KerrReturningRadiationFrameContextVerificationError(
                    "cached execution effective-kernel identity changed"
                )
        elif (
            record.live.effective_kernel is not record.live.source_kernel
            or record.live.underlying_kernel is not record.live.source_kernel
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "full source-kernel identity changed"
            )
        if object.__getattribute__(record.live.underlying_kernel, "surface") is not (
            record.live.kernel_surface
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "underlying kernel no longer owns its authenticated original surface"
            )
        _require_exact_tree(
            surface,
            record.live.kernel_surface,
            "render_surface/original_kernel_surface",
        )
        before_provider_identity = _live_identity_sha256(provider, "provider")
        before_surface_identity = _live_identity_sha256(surface, "render_surface")
        if (
            before_provider_identity.encode("ascii")
            != record.provider_live_identity_sha256.encode("ascii")
            or before_surface_identity.encode("ascii")
            != record.render_surface_live_identity_sha256.encode("ascii")
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "live provider/profile/kernel/surface tree changed after authentication"
            )
        current_snapshot = _build_snapshot(record.live, record.source_closure)
        _require_exact_tree(
            current_snapshot,
            record.snapshot,
            "authority.snapshot",
        )
        _stable_current_closure(
            record.source_closure,
            "source_closure.after_require",
        )
        after_provider_identity = _live_identity_sha256(provider, "provider")
        after_surface_identity = _live_identity_sha256(surface, "render_surface")
        if (
            after_provider_identity.encode("ascii")
            != before_provider_identity.encode("ascii")
            or after_surface_identity.encode("ascii")
            != before_surface_identity.encode("ascii")
        ):
            raise KerrReturningRadiationFrameContextVerificationError(
                "live provider or render surface changed while authority was "
                "being required"
            )
        return record.snapshot

    return (
        ValidatedReturningThermalAuthority,
        authenticate_returning_thermal_emission,
        require_authority,
    )


(
    ValidatedReturningThermalAuthority,
    authenticate_returning_thermal_emission,
    require_authority,
) = _make_authority_api()


__all__ = (
    "D20_CERTIFICATE_IMPLEMENTATION_ID",
    "IMPLEMENTATION_ID",
    "MAXIMUM_AUTHENTICATED_ANNULUS_COUNT",
    "MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL",
    "MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH",
    "MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT",
    "MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH",
    "SCIENTIFIC_STATUS",
    "SNAPSHOT_IMPLEMENTATION_ID",
    "KerrReturningRadiationFrameContextError",
    "KerrReturningRadiationFrameContextVerificationError",
    "ReturningThermalD20FluxCertificateV1",
    "ReturningThermalEmissionSnapshotV1",
    "ReturningThermalSourceClosureEntryV1",
    "ValidatedReturningThermalAuthority",
    "authenticate_returning_thermal_emission",
    "require_authority",
)
