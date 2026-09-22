from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from .state import (
    DataState,
    check_state,
    get_flux,
    get_ivar,
    get_mask,
    get_meta,
    get_state,
    is_payload,
    single_state,
    stamp_state,
)

_UNSET: Any = object()

# Instances already warned about a non-propagated companion ``ivar``.  The
# "warned once" bookkeeping lives here — never on the instance — so
# ``__call__`` keeps ``__dict__`` byte-stable and one transform can be shared
# across ``-J`` worker threads.  Weak entries drop out with their instance.
_IVAR_WARNED: "weakref.WeakSet[Any]" = weakref.WeakSet()


@dataclass
class PayloadView:
    """A normalized view over a tensor / dict payload / :class:`Payload`.

    Gives transforms one code path for flux, companion inverse variance,
    validity mask and declared processing state, and a :meth:`replace` that
    rebuilds the *same kind* of container the caller passed in.
    """

    original: Any
    flux: torch.Tensor
    ivar: torch.Tensor | None
    mask: torch.Tensor | None
    state: DataState | None
    meta: dict[str, Any]

    @property
    def is_plain(self) -> bool:
        return torch.is_tensor(self.original)

    def effective_mask(self, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Explicit ``mask=`` argument wins over the payload's own mask."""
        return mask if mask is not None else self.mask

    def replace(
        self,
        flux: torch.Tensor,
        *,
        ivar: Any = _UNSET,
        state: Any = _UNSET,
    ) -> Any:
        """Rebuild the original container with new flux (and optionally ivar)."""
        new_ivar = self.ivar if ivar is _UNSET else ivar
        if self.is_plain:
            return flux
        if isinstance(self.original, dict):
            out = dict(self.original)
            out["flux"] = flux
            if ivar is not _UNSET:
                if new_ivar is None:
                    out.pop("ivar", None)
                else:
                    out["ivar"] = new_ivar
            if state is not _UNSET:
                if state is None:
                    out.pop("state", None)
                else:
                    out["state"] = state
            return out
        # Payload container.  A state the input never declared must not be
        # fabricated here (dict payloads keep their exact shape): only an
        # explicitly requested state change replaces it.
        from .state import Payload

        return Payload(
            flux=flux,
            ivar=new_ivar,
            mask=self.mask,
            state=state if state is not _UNSET else self.original.state,
            meta=dict(self.meta),
        )


class FITSTransform:
    """Protocol for astronomy transforms with forward and inverse passes.

    Subclasses should override ``forward`` and ``inverse``.
    Calling an instance directly delegates to :meth:`forward`.

    All transforms accept an optional ``mask`` parameter
    (``torch.Tensor | None``) on both :meth:`forward` and
    :meth:`inverse`.  The mask is a boolean tensor where ``True``
    indicates a valid pixel.  Transforms that compute statistics
    (median, min, max, etc.) use the mask to exclude invalid
    pixels; pointwise transforms can safely ignore it.

    Inputs may be a bare :class:`torch.Tensor`, the
    ``{"flux", "ivar"?, "mask"?}`` dict payload emitted by
    :mod:`torchfits.data`, or a :class:`~torchfits.transforms.state.Payload`.
    Transforms that support it propagate ``ivar`` exactly through the
    operation; transforms that cannot (nonlinear stretches and clipping)
    pass it through unchanged and warn once.

    Class attributes
    ----------------
    expects : frozenset[DataState] or None
        Processing states this transform accepts. ``None`` accepts anything.
    produces : DataState or None
        State of the output payload, when the transform changes it.
    propagates_ivar : bool
        Whether ``ivar`` is transformed consistently with ``flux``.
    """

    expects: frozenset[DataState] | None = None
    produces: DataState | None = None
    propagates_ivar: bool = False

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def __call__(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        out = self.forward(x, mask=mask)
        # Keep a declared processing state accurate: without this, a payload
        # that says "stored" still says "stored" after being calibrated, and the
        # guard against applying a header-scaling transform twice never fires
        # inside a pipeline.
        return stamp_state(out, self.produces)

    def inverse_state(self) -> DataState | None:
        """State an ``inverse()`` pass restores: the one state it accepts."""
        if self.produces is None:
            return None
        return single_state(self.expects)

    def inverse_expects(self) -> frozenset[DataState] | None:
        """States an ``inverse()`` pass accepts — what :meth:`forward` produced.

        Undoing a transform must accept the state that transform produced, not
        the state it consumed, or ``t.inverse(t(payload))`` would reject its own
        output for every state-carrying payload.
        """
        if self.produces is None:
            return self.expects
        return frozenset({self.produces})

    # -- helpers for subclasses ------------------------------------------

    def view(
        self,
        x: Any,
        *,
        state: DataState | None = None,
        expects: Any = _UNSET,
    ) -> PayloadView:
        """Validate the declared state and return a :class:`PayloadView`.

        ``expects`` overrides the class-level whitelist — ``inverse()`` passes
        :meth:`inverse_expects` so undoing a transform accepts what the forward
        pass produced.
        """
        check_state(
            x,
            self.expects if expects is _UNSET else expects,
            type(self).__name__,
            state=state,
        )
        return PayloadView(
            original=x,
            flux=get_flux(x),
            ivar=get_ivar(x),
            mask=get_mask(x),
            state=get_state(x) if state is None else state,
            meta=get_meta(x),
        )

    @staticmethod
    def _as_factor(ivar: torch.Tensor, factor: Any) -> torch.Tensor:
        f = (
            factor
            if torch.is_tensor(factor)
            else torch.as_tensor(factor, dtype=ivar.dtype, device=ivar.device)
        )
        return f.to(device=ivar.device, dtype=ivar.dtype)

    @classmethod
    def scale_ivar(cls, ivar: torch.Tensor | None, factor: Any) -> torch.Tensor | None:
        """Propagate ``ivar`` when flux is multiplied by *factor*.

        Since ``var(a*x) = a**2 * var(x)``, the transformed inverse variance is
        ``ivar / factor**2``. An additive offset leaves ``ivar`` unchanged, so
        a pure offset passes ``factor=1.0``.
        """
        if ivar is None:
            return None
        f = cls._as_factor(ivar, factor)
        return ivar / (f * f)

    @classmethod
    def divide_ivar(
        cls, ivar: torch.Tensor | None, divisor: Any
    ) -> torch.Tensor | None:
        """Propagate ``ivar`` when flux is *divided* by *divisor*.

        The reciprocal of :meth:`scale_ivar`: ``ivar * divisor**2``.
        """
        if ivar is None:
            return None
        d = cls._as_factor(ivar, divisor)
        return ivar * (d * d)

    @classmethod
    def delta_ivar(cls, ivar: torch.Tensor | None, slope: Any) -> torch.Tensor | None:
        """Propagate ``ivar`` through a pointwise *nonlinear* map.

        First-order (delta-method) error propagation: since
        ``var(f(x)) ≈ (df/dx)**2 var(x)``, the transformed inverse variance is
        ``ivar / (df/dx)**2``.

        Where *slope* is non-positive, zero or non-finite the map has clamped
        or collapsed locally and is no longer injective, so the output carries
        no first-order information about the input. Those entries get
        ``ivar = 0`` (infinite variance) rather than a spuriously finite value
        — or, at a singular point, a spuriously *infinite* one.
        """
        if ivar is None:
            return None
        s = cls._as_factor(ivar, slope)
        good = torch.isfinite(s) & (s > 0)
        safe = torch.where(good, s, torch.ones_like(s))
        out = ivar / (safe * safe)
        return torch.where(good, out, torch.zeros_like(out))

    def _warn_ivar_not_propagated(self) -> None:
        """Warn once per instance that ``ivar`` is passed through unchanged."""
        if self in _IVAR_WARNED:
            return
        _IVAR_WARNED.add(self)
        warnings.warn(
            f"{type(self).__name__} is nonlinear: companion 'ivar' is passed "
            "through unchanged and no longer strictly describes the transformed "
            "flux. Drop the ivar field or carry the transform in the model.",
            UserWarning,
            stacklevel=3,
        )

    def carries_companions(self, *args: Any) -> bool:
        """True when any input carries an ``ivar`` tensor."""
        return any(is_payload(a) and get_ivar(a) is not None for a in args)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose(FITSTransform):
    """Chain transforms; ``.inverse()`` unwinds them in reverse order."""

    def __init__(self, transforms: Sequence[FITSTransform]) -> None:
        self.transforms = list(transforms)
        self.expects = self.transforms[0].expects if self.transforms else None
        self.propagates_ivar = all(
            getattr(t, "propagates_ivar", False) for t in self.transforms
        )

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int) -> FITSTransform:
        return self.transforms[idx]

    def __iter__(self) -> Iterator[FITSTransform]:
        return iter(self.transforms)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        for t in self.transforms:
            x = t(x, mask=mask)
        return x

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        for t in reversed(self.transforms):
            x = t.inverse(x, mask=mask)
        return x

    def __repr__(self) -> str:
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n])"


class AsModule(torch.nn.Module):
    """Thin ``nn.Module`` adapter around a :class:`FITSTransform`.

    Lets callers write ``torch.nn.Sequential(AsModule(pipeline), model)``.
    Only the forward pass is exposed; use ``transform.inverse`` for undo.
    """

    def __init__(self, transform: FITSTransform) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        # Delegate to __call__ (not the raw forward) so the wrapped
        # transform's state stamping still happens inside nn.Sequential
        # pipelines — otherwise a payload keeps a stale state label and the
        # double-scaling guard never fires past this adapter.
        return self.transform(x, mask=mask)


def as_module(transform: FITSTransform) -> AsModule:
    """Wrap *transform* as an :class:`AsModule` for ``nn.Sequential``."""
    return AsModule(transform)
