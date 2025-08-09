# Migration Guide: astropy.io.fits / fitsio → torchfits (Draft)

Goal: Enable nearly drop-in replacement for common workflows.

Mapping Table (to fill):

| Task | astropy.io.fits | fitsio | torchfits (planned) |
|------|-----------------|--------|---------------------|
| Open file | fits.open(path) | fitsio.FITS(path) | torchfits.open(path) |
| Read image primary | hdul[0].data | f[0].read() | torchfits.read(path).data |
| Read table | Table.read(path) | f[hdu].read() | torchfits.read(path, hdu=1) |
| Header edit | hdul[0].header['KEY']=val | f[0].write_key('KEY', val) | hdu.header['KEY']=val (write()) |
| Memory map | memmap=True | (implicit) | torchfits.open(path, mmap=True) |

Deprecation & Differences (draft):

* Lazy reading semantics differences.
* Unified tensor return vs numpy arrays.
* Error handling strategy.
