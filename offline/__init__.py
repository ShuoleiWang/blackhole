"""Offline scientific-rendering primitives.

The browser renderer remains a separate delivery product.  Modules in this
package operate on linear, invariant physical quantities before any camera or
HDR display transform.
"""

from .kerr import (
    KerrConstantsOfMotion,
    KerrKerrSchildMetric,
    KerrOblateEvent,
    KerrOblateTermination,
    KerrZamoTetrad,
    kerr_bl_vector_to_ks_cartesian,
    kerr_bl_zamo_tetrad,
    kerr_constants_of_motion,
    kerr_ks_event_to_oblate,
    kerr_oblate_event_to_ks_cartesian,
    kerr_oblate_radius_m,
    kerr_zamo_camera_ray,
    stationary_axisymmetric_constants,
)

from .radiative_transfer import (
    RadiativeTransferError,
    StepBudgetExceeded,
    StokesInvariant,
    TransferCoefficients,
    TransferDiagnostics,
    TransferIntegrationError,
    TransferResult,
    TransferSegment,
    TransferValidationError,
    propagate_source_to_observer,
)

__all__ = (
    "KerrConstantsOfMotion",
    "KerrKerrSchildMetric",
    "KerrOblateEvent",
    "KerrOblateTermination",
    "KerrZamoTetrad",
    "RadiativeTransferError",
    "StepBudgetExceeded",
    "StokesInvariant",
    "TransferCoefficients",
    "TransferDiagnostics",
    "TransferIntegrationError",
    "TransferResult",
    "TransferSegment",
    "TransferValidationError",
    "kerr_bl_vector_to_ks_cartesian",
    "kerr_bl_zamo_tetrad",
    "kerr_constants_of_motion",
    "kerr_ks_event_to_oblate",
    "kerr_oblate_event_to_ks_cartesian",
    "kerr_oblate_radius_m",
    "kerr_zamo_camera_ray",
    "propagate_source_to_observer",
    "stationary_axisymmetric_constants",
)
