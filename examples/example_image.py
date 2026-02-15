"""
Example script demonstrating basic image reading and header access.
"""


def main():
    # Read image data (returns torch.Tensor on CPU by default)
    print("Reading image...")
    # Simulated data for example - in real usage, provide a path to a FITS file
    # data = torchfits.read("path/to/image.fits")

    # Read image and header together
    # data, header = torchfits.read("path/to/image.fits", return_header=True)

    print("Example completed (simulated).")


if __name__ == "__main__":
    main()
