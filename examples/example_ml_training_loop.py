"""Example: a real ML pipeline end to end, from FITS to a training step.

This is the "does it actually work for ML?" example. It builds a tiny survey
on disk (multi-band cutouts, 1D spectra, an IFU cube, a catalog), loads it
through ``torchfits.data``, preprocesses it with ``torchfits.transforms``, and
trains a small model — exercising the parts of the library an ML user relies
on:

1. ``discover_bands`` + ``FitsImageDataset.from_bands`` — a multi-band dataset
   built from extension names, with IVAR/DQ companions decoded for you.
2. The transform state contract — instrumentation that the reader already
   applied is not applied twice, and an already continuum-normalized spectrum
   is left alone instead of being silently re-normalized.
3. Mask/IVAR awareness — DQ bitfields become validity masks, and the mask
   reaches the loss instead of being dropped on the floor.
4. Spectra, IFU cubes and tables, not just images.
5. A training loop whose loss goes down.

Runs on CPU in a few seconds; no network access.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import DataLoader

from torchfits.data import (
    FitsCubeDataset,
    FitsImageDataset,
    FitsSpectrumDataset,
    FitsTableIterableDataset,
    discover_bands,
    make_loader,
)
from torchfits.transforms import (
    ArcsinhStretch,
    Compose,
    DataState,
    DataStateError,
    MeshBackgroundSubtract,
    Payload,
    SigmaNormalize,
    apply_mask,
    estimate_background,
    mask_from_dq,
)

N_IMAGES = 12
N_SPEC = 6
SIZE = 32
NWAVE = 64
SEED = 7


# ---------------------------------------------------------------------------
# Demo data: a survey-like mix of products
# ---------------------------------------------------------------------------


def _image_cube(rng: np.random.Generator, bright: bool) -> dict[str, np.ndarray]:
    """One 3-band cutout: a Gaussian source on a drifting, noisy sky."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    cy, cx = rng.integers(8, SIZE - 8, size=2)
    amp = 2000.0 if bright else 200.0
    src = amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5**2))
    # A background gradient across the frame — why a single global median is
    # not enough, and why MeshBackgroundSubtract exists.
    gradient = 0.4 * x
    bands = {}
    for name, sky in (("G", 100.0), ("R", 120.0), ("Z", 140.0)):
        flux = (sky + gradient + src + rng.normal(0, 5.0, (SIZE, SIZE))).astype(
            np.float32
        )
        bands[name] = flux
        bands[f"{name}_IVAR"] = np.full((SIZE, SIZE), 1.0 / 25.0, np.float32)
    # A few flagged pixels per band, in the two flavours real pipelines see:
    # a fatal defect (bit 2) and a benign informational flag (bit 10).
    dq = np.zeros((SIZE, SIZE), np.int16)
    dq[rng.integers(0, SIZE, 3), rng.integers(0, SIZE, 3)] = 4
    dq[rng.integers(0, SIZE, 3), rng.integers(0, SIZE, 3)] = 1024
    for name in ("G", "R", "Z"):
        bands[f"{name}_DQ"] = dq.copy()
    return bands


def write_images(root: str) -> list[str]:
    rng = np.random.default_rng(SEED)
    paths = []
    for i in range(N_IMAGES):
        path = os.path.join(root, f"cutout_{i:03d}.fits")
        hdus = [fits.PrimaryHDU()]
        for name, value in _image_cube(rng, bright=bool(i % 2)).items():
            hdu = fits.ImageHDU(value, name=name)
            if not name.endswith(("_IVAR", "_DQ")):
                hdu.header["ZP"] = 26.0
                hdu.header["EXPTIME"] = 100.0
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path, overwrite=True)
        paths.append(path)
    return paths


def write_spectra(root: str) -> list[str]:
    """Continuum + an absorption line, with IVAR, wavelength and a DQ column."""
    rng = np.random.default_rng(SEED + 1)
    wave = np.linspace(4000.0, 7000.0, NWAVE).astype(np.float32)
    paths = []
    for i in range(N_SPEC):
        path = os.path.join(root, f"spec_{i:03d}.fits")
        continuum = 1.0 + 0.3 * np.sin(wave / 900.0)
        depth = 0.2 + 0.6 * (i / N_SPEC)
        line = depth * np.exp(-((wave - 5200.0) ** 2) / (2 * 120.0**2))
        flux = (continuum - line + rng.normal(0, 0.02, NWAVE)).astype(np.float32)
        ivar = np.full(NWAVE, 1.0 / 0.02**2, np.float32)
        dq = np.zeros(NWAVE, np.int32)
        dq[rng.integers(0, NWAVE, 2)] = 2
        cols = [
            fits.Column(name="FLUX", format="E", array=flux),
            fits.Column(name="IVAR", format="E", array=ivar),
            fits.Column(name="WAVE", format="E", array=wave),
            fits.Column(name="DQ", format="J", array=dq),
        ]
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
            path, overwrite=True
        )
        paths.append(path)
    return paths


def write_cube(root: str) -> str:
    """A small IFU-style datacube: (wavelength, spaxel_y, spaxel_x)."""
    rng = np.random.default_rng(SEED + 2)
    cube = rng.normal(10.0, 1.0, (24, 16, 16)).astype(np.float32)
    cube[10:14, 6:10, 6:10] += 25.0  # a bright blob in a few channels
    path = os.path.join(root, "ifu_cube.fits")
    fits.PrimaryHDU(cube).writeto(path, overwrite=True)
    return path


def write_catalog(root: str) -> str:
    rng = np.random.default_rng(SEED + 3)
    n = 64
    cols = [
        fits.Column(
            name="MAG", format="E", array=rng.normal(21, 1, n).astype(np.float32)
        ),
        fits.Column(
            name="Z", format="E", array=rng.uniform(0, 1, n).astype(np.float32)
        ),
        fits.Column(
            name="SNR", format="E", array=rng.uniform(3, 40, n).astype(np.float32)
        ),
    ]
    path = os.path.join(root, "catalog.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return path


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TinyCNN(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU(),
            # Max, not average: a point source is a few bright pixels, and
            # averaging them over the frame is exactly how you lose the signal.
            torch.nn.AdaptiveMaxPool2d(1),
        )
        self.head = torch.nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_bands(paths: list[str]) -> FitsImageDataset:
    print("\n1. Band discovery — build a dataset from extension names")
    for info in discover_bands(paths[0]):
        print(
            f"   {info.name:<7} role={info.role:<10} shape={info.shape} "
            f"zp={info.zeropoint}"
        )

    labels = [i % 2 for i in range(len(paths))]
    dataset = FitsImageDataset.from_bands(paths, labels=labels)
    print(f"   -> {dataset}")
    print(f"   zeropoints: {dataset.band_zeropoints()}")
    print(f"   decoded DQ? {dataset.mask_is_dq}")

    payload, label = dataset[0]
    g = next(b for b in discover_bands(paths[0]) if b.name == "G")
    print(
        f"   payload: flux{tuple(payload['flux'].shape)} "
        f"ivar{tuple(payload['ivar'].shape)} "
        f"mask{tuple(payload['mask'].shape)} dtype={payload['mask'].dtype} "
        f"label={int(label)}"
    )
    zeropoint = g.zeropoint or 0.0
    print(
        f"   photometry: ZP={zeropoint:.1f} -> counts x {g.flux_scale():.3g} = flux, "
        f"or {(-2.5 * np.log10(100.0) + zeropoint):.2f} mag for a 100-count source"
    )
    return dataset


def section_state_contract() -> None:
    print("\n2. The state contract — calibration is never applied twice")
    counts = torch.ones(8, 8) * 500.0
    continuum_normalized = Payload(flux=counts, state=DataState.CONTINUUM_NORMALIZED)
    try:
        SigmaNormalize()(continuum_normalized)
    except DataStateError as exc:
        print(f"   refused, as it should be:\n     {exc}")
    # The physical-state payload is the one normalizers are for.
    physical = Payload(flux=counts, state=DataState.PHYSICAL)
    out = SigmaNormalize()(physical)
    print(f"   state after SigmaNormalize: {out.state.value}")


def build_pipeline() -> Compose:
    """Sky subtraction, then unit-variance scaling: the model-input recipe."""
    return Compose(
        [
            # SExtractor-style tiled sky: handles the frame-to-frame gradient.
            MeshBackgroundSubtract(mesh=(4, 4), weighted=True),
            # Per-channel robust scale, zero-preserving so band ratios survive.
            SigmaNormalize(stat="mad"),
        ]
    )


def section_preprocessing(dataset: FitsImageDataset) -> None:
    print("\n3. Preprocessing — mask- and IVAR-aware, and checked numerically")
    payload, _ = dataset[0]
    pipeline = build_pipeline()
    processed = pipeline(payload)

    flux, mask = payload["flux"], payload["mask"]
    out = processed["flux"]
    print(
        f"   raw    : median={float(estimate_background(flux, mask=mask)[0].mean()):8.2f}"
        f"  mad={float(estimate_background(flux, mask=mask)[1].mean()):6.2f}"
    )
    print(
        f"   preproc: median={float(estimate_background(out, mask=mask)[0].mean()):8.4f}"
        f"  mad={float(estimate_background(out, mask=mask)[1].mean()):6.4f}"
    )
    print(f"   valid pixels kept by the DQ mask: {int(mask.sum())}/{mask.numel()}")

    # The stretch is the display/value-range step; with propagate_ivar it keeps
    # the companion uncertainty consistent with the stretched flux.
    stretch = ArcsinhStretch(a=0.1, propagate_ivar=True)
    stretched = Compose([SigmaNormalize(), stretch])(payload)
    ratio = (stretched["ivar"] / payload["ivar"]).mean()
    print(
        f"   ArcsinhStretch(propagate_ivar=True): ivar scaled by "
        f"{float(ratio):.3e} on average, all finite="
        f"{bool(torch.isfinite(stretched['ivar']).all())}"
    )


def section_spectra(spectra: list[str]) -> None:
    print("\n4. Spectra — wavelength, DQ mask and labels from a table")
    dataset = FitsSpectrumDataset(
        spectra,
        hdu=1,
        column="FLUX",
        ivar_column="IVAR",
        mask_column="DQ",
        mask_is_dq=True,
        wavelength_column="WAVE",
        labels=[i % 2 for i in range(len(spectra))],
    )
    payload, label = dataset[0]
    print(
        f"   flux{tuple(payload['flux'].shape)} "
        f"wavelength{tuple(payload['wavelength'].shape)} "
        f"valid={int(payload['mask'].sum())}/{payload['mask'].numel()} "
        f"label={int(label)}"
    )
    wave = payload["wavelength"]
    print(f"   wavelength range: {float(wave.min()):.1f} .. {float(wave.max()):.1f} A")

    # A survey spectrum is usually already continuum-normalized before it
    # reaches you. Say so and the normalizers leave it alone instead of
    # dividing out the common scale the reduction established.
    declared = Payload(flux=payload["flux"], state=DataState.CONTINUUM_NORMALIZED)
    try:
        SigmaNormalize()(declared)
    except DataStateError:
        print(
            "   continuum-normalized spectrum -> DataStateError "
            "(a normalizer refuses to re-scale it)"
        )


def section_ifu(cube_path: str) -> None:
    print("\n5. IFU cube — select a spectral window instead of the whole cube")
    full = FitsCubeDataset(cube_path, hdu=0)
    window = FitsCubeDataset(cube_path, hdu=0, spectral_slice=(8, 16))
    print(f"   full   : {tuple(full[0][0].shape)}")
    print(f"   window : {tuple(window[0][0].shape)}  (channels 8..15)")


def section_tables(catalog: str) -> None:
    print("\n6. Catalog — contiguous row sharding, streamed as tensors")
    seen: list[float] = []
    for rank in range(2):
        batches = list(
            FitsTableIterableDataset(
                catalog,
                hdu=1,
                columns=["MAG", "Z", "SNR"],
                batch_size=16,
                rank=rank,
                world_size=2,
                as_batches=True,
            )
        )
        rows = sum(chunk["MAG"].numel() for chunk in batches)
        print(
            f"   rank {rank}: {len(batches)} tensor batches -> {rows} rows, "
            f"columns={sorted(batches[0])}"
        )
        for row in FitsTableIterableDataset(
            catalog, hdu=1, columns=["MAG"], rank=rank, world_size=2
        ):
            seen.append(round(float(row["MAG"]), 5))
    print(
        f"   both shards cover every row once: {len(seen)} rows, "
        f"distinct={len(set(seen))}"
    )


def _model_input(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask-aware, heavy-tail-safe input for the model.

    The DQ mask is applied first so flagged pixels cannot drive the gradient,
    then the dynamic range is clamped: a bright source is hundreds of sigma
    above the sky, and an unclamped tensor lets those few pixels dominate the
    first layer.
    """
    return apply_mask(payload["flux"], payload["mask"], fill=0.0).clamp(-5.0, 25.0)


def section_training(dataset: FitsImageDataset) -> None:
    print("\n7. Training loop")
    torch.manual_seed(SEED)

    train_ds = FitsImageDataset.from_bands(
        dataset.files,
        labels=[i % 2 for i in range(len(dataset.files))],
        transform=build_pipeline(),
    )
    loader = make_loader(train_ds, batch_size=4, num_workers=0, pin_memory=False)

    model = TinyCNN(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    for epoch in range(8):
        total = count = correct = 0
        for payload, labels in loader:
            x = _model_input(payload)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            count += len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
        losses.append(total / count)
        print(f"   epoch {epoch}: loss={losses[-1]:.4f} accuracy={correct / count:.2f}")

    accuracy = _evaluate(model, train_ds)
    print(f"   loss {losses[0]:.4f} -> {losses[-1]:.4f}; final accuracy={accuracy:.2f}")
    if not losses[-1] < losses[0]:
        raise AssertionError(
            f"training loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


@torch.no_grad()
def _evaluate(model: TinyCNN, dataset: FitsImageDataset) -> float:
    loader = DataLoader(dataset, batch_size=4)
    correct = total = 0
    for payload, labels in loader:
        correct += int((model(_model_input(payload)).argmax(dim=1) == labels).sum())
        total += len(labels)
    return correct / max(total, 1)


def main() -> None:
    root = tempfile.mkdtemp(prefix="torchfits_ml_")
    try:
        images = write_images(root)
        spectra = write_spectra(root)
        cube = write_cube(root)
        catalog = write_catalog(root)

        dataset = section_bands(images)
        section_state_contract()
        section_preprocessing(dataset)
        section_spectra(spectra)
        section_ifu(cube)
        section_tables(catalog)
        section_training(dataset)

        # The library's own mask helper is useful standalone too.
        dq = torch.tensor([[0, 4], [1024, 0]], dtype=torch.int16)
        print(
            "\n8. mask_from_dq: clean frame -> all valid, "
            f"bad_bits=[2] -> {mask_from_dq(dq, bad_bits=[2]).tolist()}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
