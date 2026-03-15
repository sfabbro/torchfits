import pytest
import numpy as np
from torchfits.hdu import _safe_eval


def test_safe_eval_allowed():
    assert _safe_eval("np.sin(0)", {}, np) == 0.0
    assert _safe_eval("np.cos(0)", {}, np) == 1.0
    assert _safe_eval("np.exp(0)", {}, np) == 1.0
    assert _safe_eval("np.pi", {}, np) == np.pi


def test_safe_eval_blocked():
    with pytest.raises(
        AttributeError, match="Attribute 'loadtxt' on 'np' is not allowed"
    ):
        _safe_eval("np.loadtxt('/etc/passwd')", {}, np)

    with pytest.raises(AttributeError, match="Attribute 'core' on 'np' is not allowed"):
        _safe_eval("np.core.multiarray", {}, np)

    with pytest.raises(AttributeError, match="Attribute 'lib' on 'np' is not allowed"):
        _safe_eval("np.lib.utils.safe_eval", {}, np)
