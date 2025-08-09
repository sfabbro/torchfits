# Quickstart (Draft)

Install (development environment managed by pixi):

```bash
pixi run build-dev
```

Read an image HDU directly into a tensor:

```python
import torchfits as tf
hdu = tf.read("examples/basic_example.fits")
img = hdu.data  # torch.Tensor
print(img.shape, img.dtype)
```

Load a table and convert to torch-frame (if installed):

```python
tab = tf.read("examples/table_example.fits", hdu=1)
df = tab.to_torch_frame()  # stypes inferred automatically (target ≥95% accuracy)
```

Next: see `guide/data_access.md` for images, tables, cubes, MEFs.
