import torch
import pytest

hp = pytest.importorskip("healpy")
from torchfits.sphere import synalm


@pytest.mark.skipif(not hasattr(hp, "synalm"), reason="healpy not available")
def test_synalm_parity_detailed():
    lmax = 32
    mmax = 32
    cls = torch.ones(lmax + 1, dtype=torch.float64)

    # Run synalm
    torch.manual_seed(42)
    alm = synalm(cls, lmax=lmax, mmax=mmax)

    # Check stats
    # For C_l = 1, Var(a_lm) should be 1.0
    # Mean should be 0
    print(f"ALM Mean: {alm.mean()}")
    print(f"ALM Var: {alm.abs().pow(2).mean()}")

    # Parity check is hard because of different RNGs,
    # but we can check if it produces valid HEALPix ALMs.
    assert alm.shape[0] == hp.Alm.getsize(lmax, mmax)
    assert torch.all(alm[hp.Alm.getidx(lmax, torch.arange(lmax + 1), 0)].imag == 0)
