import importlib

import torch


def test_dlpack_roundtrip_cpu():
    # echo_tensor is a nanobind test helper on _C, not the sealed _cpp façade.
    m = importlib.import_module("torchfits._C")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = m.echo_tensor(t)
    # CPU tensors should share the same storage pointer (zero-copy round-trip)
    assert out.data_ptr() == t.data_ptr()


def test_echo_tensor_survives_source_free():
    # Lifetime pin (r5c-11): the echoed tensor must keep the source storage
    # alive after the Python source tensor is garbage-collected.
    import gc
    import weakref

    m = importlib.import_module("torchfits._C")

    def build():
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        return m.echo_tensor(t), weakref.ref(t.untyped_storage())

    out, ref = build()
    gc.collect()
    assert ref() is not None, "echo_tensor result dropped the source storage"
    assert out.data_ptr() != 0
    assert float(out.sum().item()) == float(sum(range(12)))
