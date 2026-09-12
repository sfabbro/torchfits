"""Data-state contract for transforms (stored vs physical, normalized, …).

A transform that silently re-applies calibration the reader already applied is
a correctness bug: ``read_tensor`` returns *physical* values (BSCALE/BZERO
applied) and ``table.read_torch`` returns physical column values
(TSCAL/TZERO applied, TNULL already NaN). Feeding that into
:class:`~torchfits.transforms.FITSHeaderScale.forward` scales it twice.

This module makes the assumption explicit:

- :class:`DataState` names the processing stage a tensor payload is in.
- Transforms declare ``expects`` / ``produces``; when a payload *declares* a
  conflicting state the transform raises :class:`DataStateError` with an
  actionable message instead of silently corrupting the data.
- :class:`Payload` is an optional typed container carrying ``flux``,
  ``ivar``, ``mask``, ``state`` and free-form ``meta``. It is interchangeable
  with the plain ``{"flux", "ivar"?, "mask"?}`` dict payload the Dataset layer
  already emits, so no existing call site has to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Any, Mapping

import torch

__all__ = [
    "DataState",
    "DataStateError",
    "Payload",
    "as_state",
    "calibration_state",
    "SCALABLE",
    "check_state",
    "get_flux",
    "get_ivar",
    "get_mask",
    "get_meta",
    "get_state",
    "is_payload",
    "set_flux",
    "set_state",
    "stamp_state",
    "single_state",
]


class DataState(str, Enum):
    """Processing stage of a flux payload.

    ``STORED``
        Raw FITS storage codes: BSCALE/BZERO (or TSCAL/TZERO, TNULL) **not**
        applied. Produced by ``read_tensor(..., raw_scale=True)``.
    ``PHYSICAL``
        Calibrated physical values (counts → flux / surface brightness).
        Produced by the default ``read_tensor`` / ``table.read_torch`` path.
    ``CONTINUUM_NORMALIZED``
        Spectra divided by their continuum: dimensionless, continuum ≈ 1.
        torchfits never *produces* this — it only respects it.
    ``NORMALIZED``
        Dimensionless model input (e.g. ``[0, 1]`` or zero-median / unit-scale).
    """

    STORED = "stored"
    PHYSICAL = "physical"
    CONTINUUM_NORMALIZED = "continuum_normalized"
    NORMALIZED = "normalized"


SCALABLE: frozenset[DataState] = frozenset(
    {DataState.STORED, DataState.PHYSICAL, DataState.NORMALIZED}
)
"""States a flux-scaling transform accepts.

``CONTINUUM_NORMALIZED`` is deliberately absent: dividing an already
continuum-normalized spectrum by another per-spectrum statistic destroys the
common scale the normalization was there to establish, so the normalizers
refuse it instead of silently redoing the work.
"""


class DataStateError(ValueError):
    """A transform was handed a payload in an incompatible processing state."""


@dataclass
class Payload:
    """Flux tensor plus its uncertainty, validity mask and processing state."""

    flux: torch.Tensor
    ivar: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    state: DataState | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the equivalent plain-dict payload form."""
        out: dict[str, Any] = {"flux": self.flux}
        if self.ivar is not None:
            out["ivar"] = self.ivar
        if self.mask is not None:
            out["mask"] = self.mask
        if self.state is not None:
            out["state"] = self.state
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    @classmethod
    def coerce(cls, x: Any) -> "Payload":
        """Build a :class:`Payload` from a Payload, dict, or bare tensor."""
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            flux = x.get("flux")
            if not torch.is_tensor(flux):
                raise TypeError(
                    f"dict payload requires a torch.Tensor 'flux' field, got {type(flux)}"
                )
            ivar = x.get("ivar")
            mask = x.get("mask")
            meta = x.get("meta")
            return cls(
                flux=flux,
                ivar=ivar if torch.is_tensor(ivar) else None,
                mask=mask if torch.is_tensor(mask) else None,
                state=as_state(x.get("state")),
                meta=dict(meta) if isinstance(meta, Mapping) else {},
            )
        if torch.is_tensor(x):
            return cls(flux=x)
        raise TypeError(
            f"expected torch.Tensor, dict payload, or Payload, got {type(x)}"
        )


def as_state(value: Any) -> DataState | None:
    """Coerce ``None`` / ``str`` / :class:`DataState` to a state (or None)."""
    if value is None:
        return None
    if isinstance(value, DataState):
        return value
    try:
        return DataState(str(value))
    except ValueError as exc:
        valid = ", ".join(s.value for s in DataState)
        raise ValueError(
            f"unknown data state {value!r}; expected one of: {valid}"
        ) from exc


def is_payload(x: Any) -> bool:
    """True when *x* carries companion data (Payload or ``{"flux": tensor}``)."""
    if isinstance(x, Payload):
        return True
    return isinstance(x, dict) and torch.is_tensor(x.get("flux"))


def calibration_state(
    header: Mapping[str, Any], *, raw_scale: bool = False
) -> DataState:
    """State of data produced by the reader for *header*.

    The default reader applies BSCALE/BZERO (and TSCAL/TZERO for tables), so
    its output is :attr:`DataState.PHYSICAL`; ``raw_scale=True`` skips that and
    returns :attr:`DataState.STORED`.
    """
    return DataState.STORED if raw_scale else DataState.PHYSICAL


def check_state(
    x: Any,
    expects: frozenset[DataState] | None,
    transform_name: str,
    *,
    state: DataState | None = None,
) -> DataState | None:
    """Validate a payload's declared state against *expects*.

    A bare tensor carries no state and is never rejected (the caller may know
    something the container does not). A *declared* state that is not accepted
    raises :class:`DataStateError` naming the likely cause.
    """
    actual = state if state is not None else get_state(x)
    if actual is None or expects is None or actual in expects:
        return actual
    accepted = ", ".join(sorted(s.value for s in expects))
    hint = ""
    if actual is DataState.CONTINUUM_NORMALIZED:
        hint = (
            "  Continuum-normalized spectra are already dimensionless and share "
            "one flux scale; re-normalizing them would discard that. Feed them "
            "to the model as-is, or clear the state if you really mean to "
            "rescale them."
        )
    elif DataState.STORED in expects:
        hint = (
            "  The default reader returns *physical* values already "
            "(BSCALE/BZERO applied). Read with raw_scale=True for stored "
            "codes, or drop the header-scaling transform."
        )
    raise DataStateError(
        f"{transform_name} expects state [{accepted}], but the payload declares "
        f"{actual.value!r}.{hint}"
    )


def stamp_state(x: Any, state: DataState | None) -> Any:
    """Set *state* on a payload that already declares one.

    Payloads that never opted into the contract (a bare tensor, or a dict with
    no ``state`` field) are returned unchanged, so adding state tracking does
    not reshape existing inputs. A payload that *did* declare a state keeps it
    accurate as it moves through a pipeline — which is what lets the guard in
    :func:`check_state` catch a transform applied twice.
    """
    if state is None or get_state(x) is None:
        return x
    return set_state(x, state)


def single_state(states: frozenset[DataState] | None) -> DataState | None:
    """The one state in *states*, or ``None`` when it is ambiguous/absent."""
    if states is None or len(states) != 1:
        return None
    return next(iter(states))


def get_flux(x: Any) -> torch.Tensor:
    """Extract the flux tensor from a tensor / dict payload / :class:`Payload`."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, Payload):
        return x.flux
    if isinstance(x, dict):
        flux = x.get("flux")
        if torch.is_tensor(flux):
            return flux
        keys = ", ".join(sorted(map(str, x.keys()))) or "(empty)"
        raise TypeError(
            "dict payload requires a torch.Tensor 'flux' field "
            f"(keys present: {keys}). Multi-arm spectrum dicts must be "
            "flattened (layout='stack'/'concat') before transforms run."
        )
    raise TypeError(f"expected torch.Tensor, dict payload, or Payload, got {type(x)}")


def get_ivar(x: Any) -> torch.Tensor | None:
    """Return the companion inverse-variance tensor, if any."""
    if isinstance(x, Payload):
        return x.ivar
    if isinstance(x, dict):
        ivar = x.get("ivar")
        return ivar if torch.is_tensor(ivar) else None
    return None


def get_mask(x: Any) -> torch.Tensor | None:
    """Return the payload's own boolean validity mask, if any (True = valid)."""
    if isinstance(x, Payload):
        return x.mask
    if isinstance(x, dict):
        mask = x.get("mask")
        return mask if torch.is_tensor(mask) else None
    return None


def get_meta(x: Any) -> dict[str, Any]:
    """Return payload metadata (empty dict for bare tensors)."""
    if isinstance(x, Payload):
        return x.meta
    if isinstance(x, dict):
        meta = x.get("meta")
        return dict(meta) if isinstance(meta, Mapping) else {}
    return {}


def get_state(x: Any) -> DataState | None:
    """Return the payload's declared processing state, if any."""
    if isinstance(x, Payload):
        return x.state
    if isinstance(x, dict):
        return as_state(x.get("state"))
    return None


def set_flux(x: Any, flux: torch.Tensor) -> Any:
    """Return *x* with its flux replaced, preserving container shape."""
    if isinstance(x, Payload):
        return _dc_replace(x, flux=flux)
    if isinstance(x, dict):
        out = dict(x)
        out["flux"] = flux
        return out
    return flux


def set_state(x: Any, state: DataState | None) -> Any:
    """Return *x* with its declared state replaced (no-op for bare tensors)."""
    if isinstance(x, Payload):
        return _dc_replace(x, state=state)
    if isinstance(x, dict):
        out = dict(x)
        if state is None:
            out.pop("state", None)
        else:
            out["state"] = state
        return out
    return x
