import importlib

import torch


def test_dlpack_roundtrip_cpu():
    m = importlib.import_module("torchfits.cpp")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = m.echo_tensor(t)
    # CPU tensors should share the same storage pointer (zero-copy round-trip)
    assert out.data_ptr() == t.data_ptr()
