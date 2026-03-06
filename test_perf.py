import timeit
import torch


def _old_dist(r_ra, r_dec):
    return r_ra**2 + r_dec**2


def _new_dist(r_ra, r_dec):
    return torch.square(r_ra) + torch.square(r_dec)


setup = "import torch; r_ra = torch.rand(100000); r_dec = torch.rand(100000)"
print("old:", timeit.timeit("r_ra**2 + r_dec**2", setup=setup, number=10000))
print(
    "new:",
    timeit.timeit(
        "torch.square(r_ra) + torch.square(r_dec)", setup=setup, number=10000
    ),
)
print("mul:", timeit.timeit("r_ra * r_ra + r_dec * r_dec", setup=setup, number=10000))
