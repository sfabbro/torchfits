import os

import fsspec  # Import fsspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torchfits

# --- Data Download and Caching ---

# No need of download function now

# --- PyTorch Dataset ---


class SDSSDataset(Dataset):
    def __init__(
        self, file_list, label_map, cache_capacity=0, device="cpu"
    ):  # Add cache and device
        self.file_list = file_list
        self.label_map = label_map  # Dictionary mapping class strings to integer labels
        self.cache_capacity = cache_capacity  # Add cache capacity
        self.device = device  # Add device

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        try:
            # Read the spectrum data (flux) from the table HDU
            # Try HDU 2 first (our table), then fall back to HDU 1
            try:
                data, header = torchfits.read(
                    filename,
                    hdu=2,
                    cache_capacity=self.cache_capacity,
                    device=self.device,
                )
                if hasattr(data, "keys") and "flux" in data:
                    flux = data["flux"]
                else:
                    # Fall back to HDU 1 as image data
                    flux, _ = torchfits.read(
                        filename,
                        hdu=1,
                        cache_capacity=self.cache_capacity,
                        device=self.device,
                    )
            except Exception:
                # Fall back to HDU 1 as image data
                flux, _ = torchfits.read(
                    filename,
                    hdu=1,
                    cache_capacity=self.cache_capacity,
                    device=self.device,
                )

            # Get the class from header
            try:
                header = torchfits.get_header(filename, hdu=0)
                obj_class = header.get("CLASS", "UNKNOWN")
            except Exception:
                # Fallback using astropy fits
                obj_class = fits.getval(filename, "CLASS", ext=0)

            label = self.label_map.get(
                obj_class.strip(), 0
            )  # Convert string to integer label, default to 0

            # Normalize the flux (simple example)
            if flux is not None and torch.max(flux) > 0:
                flux = flux / torch.max(flux)
            else:
                # Create dummy flux if reading failed
                flux = torch.ones(3000, dtype=torch.float32)

            return flux, torch.tensor(label, dtype=torch.long)  # Return also as tensor

        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            # Return None if there's an error.  The collate_fn will handle it.
            return None, None

    def get_wavelength(self, idx):
        """
        Get wavelengths, for plotting proposes
        """
        filename = self.file_list[idx]
        try:
            # Try to read the table HDU with wavelength data
            data, header = torchfits.read(
                filename, hdu=2, cache_capacity=self.cache_capacity, device=self.device
            )
            # For table data, torchfits returns a dictionary-like object
            if hasattr(data, "keys") and "loglam" in data:
                return data["loglam"]
            else:
                # Fallback: create synthetic wavelength array
                flux_data, _ = torchfits.read(
                    filename,
                    hdu=1,
                    cache_capacity=self.cache_capacity,
                    device=self.device,
                )
                n_points = flux_data.shape[0] if flux_data is not None else 3000
                return torch.linspace(np.log10(3800), np.log10(9200), n_points)
        except Exception as e:
            # Fallback: create synthetic wavelength array
            print(f"Warning: Could not read wavelength data for {filename}: {e}")
            return torch.linspace(np.log10(3800), np.log10(9200), 3000)


# --- Model (Simple 1D CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(
            1, 16, kernel_size=5, stride=2, padding=2
        )  # in_channels = 1
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_size // 4), 128)  # Adjust based on input_size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: [B, C, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Collate Function (to handle potential None values) ---


def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None values
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.default_collate(batch)


# --- Main Script ---


def main():
    # Instead of base_url
    fs_params = {  # Example using HTTPS.  Could also use s3://, gcs://, etc.
        "protocol": "https",
        "host": "dr17.sdss.org",
        "path": "sas/dr17/sdss/spectro/redux/v5_13_2/spectra/lite",
    }
    data_dir = "data_sdss"
    os.makedirs(data_dir, exist_ok=True)

    # Example plate/mjd/fiber combinations (replace with a larger sample if desired)
    spectra_to_download = [
        (266, 51630, 3),  # QSO
        (266, 51630, 4),  # GALAXY
        (266, 51630, 6),  # STAR
        (334, 51689, 5),  # STAR
    ]

    # Download files (using fsspec-constructed URLs) and create file list.
    file_list = []
    for plate, mjd, fiberid in spectra_to_download:
        filename = f"spec-{plate:04}-{mjd}-{fiberid:04}.fits"
        filepath = os.path.join(data_dir, filename)

        # Construct the URL directly using modern approach
        base_url = f"{fs_params['protocol']}://{fs_params['host']}"
        file_path = f"{fs_params['path']}/{plate:04}/{filename}"
        url = f"{base_url}/{file_path}"

        if not os.path.exists(filepath):
            print(f"Downloading {url}...")
            try:
                with (
                    fsspec.open(url, mode="rb") as f_remote,
                    open(filepath, "wb") as f_local,
                ):
                    f_local.write(f_remote.read())  # Read and store locally
                print(f"Successfully downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                continue  # If cannot download, continue
        else:
            print(f"File already exists: {filepath}")

        # Only add to file_list if the file actually exists
        if os.path.exists(filepath):
            file_list.append(filepath)  # Append the *local* file path

    # If no files were downloaded successfully, create dummy data for demonstration
    if not file_list:
        print(
            "\nNo SDSS files could be downloaded. Creating dummy spectral data for demonstration..."
        )
        for i, (plate, mjd, fiberid) in enumerate(spectra_to_download):
            filename = f"spec-{plate:04}-{mjd}-{fiberid:04}.fits"
            filepath = os.path.join(data_dir, filename)

            # Create dummy spectrum data
            import numpy as np_local
            from astropy.io import fits
            from astropy.table import Table

            # Create a dummy spectrum with 3000 wavelength points (typical for SDSS)
            flux = np_local.random.normal(1.0, 0.1, 3000).astype(np_local.float32)
            loglam = np_local.linspace(
                np_local.log10(3800), np_local.log10(9200), 3000
            ).astype(
                np_local.float64
            )  # Log wavelength

            # Create HDUs mimicking SDSS structure
            primary_hdu = fits.PrimaryHDU()

            # Add classification based on index
            classes = ["STAR", "GALAXY", "QSO", "STAR"]
            class_name = classes[i % len(classes)]
            primary_hdu.header["CLASS"] = class_name
            primary_hdu.header["SUBCLASS"] = class_name

            # Create data HDU with flux (as image for now)
            data_hdu = fits.ImageHDU(flux, name="COADD")

            # Create a table HDU with flux and wavelength data
            table_data = Table()
            table_data["flux"] = flux
            table_data["loglam"] = loglam
            table_hdu = fits.BinTableHDU(table_data, name="COADD_TABLE")

            # Create HDU list and save
            hdul = fits.HDUList([primary_hdu, data_hdu, table_hdu])
            hdul.writeto(filepath, overwrite=True)

            file_list.append(filepath)
            print(f"Created dummy spectrum: {filename} (class: {class_name})")

    # Define class labels
    label_map = {"STAR": 0, "GALAXY": 1, "QSO": 2}
    num_classes = len(label_map)

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Dataset and DataLoader.  Demonstrate different cache sizes.
    print("--- Training with cache_capacity=10 ---")
    dataset = SDSSDataset(
        file_list, label_map, cache_capacity=10, device=str(device)
    )  # Pass cache and device
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    # Get wavelengths, assuming they are the same
    wavelengths = np.power(10, dataset.get_wavelength(0).numpy())

    # Determine input size from the first spectrum
    # We need to check for None in case the first item failed to load.
    first_item = dataset[0]
    if first_item is None:
        raise RuntimeError(
            "First item in dataset is None. Check for file download issues."
        )
    input_size = first_item[0].shape[0]

    # Initialize model, loss, and optimizer
    model = SimpleCNN(input_size, num_classes).to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop (Simplified) ---
    num_epochs = 3  # Reduced for faster demonstration
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            # Skip if the batch is empty (due to None values)
            if inputs.numel() == 0:
                continue

            # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    print("Training finished!")

    # --- Basic Evaluation (on the training data - for demonstration only!) ---
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations during evaluation
        for inputs, labels in dataloader:
            # Skip if the batch is empty (due to None values)
            if inputs.numel() == 0:
                continue

            # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the training set: {100 * correct / total:.2f}%")

    # --- Plotting a Spectrum (Example) ---
    try:
        import matplotlib.pyplot as plt

        # Get one spectrum for plotting
        spectrum, label = dataset[0]  # Get the first spectrum
        if spectrum is not None:
            spectrum = spectrum.numpy()
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, spectrum)
            plt.xlabel("Wavelength (Angstroms)")
            plt.ylabel("Flux (normalized)")
            plt.title(
                f"SDSS Spectrum (Class: {list(label_map.keys())[list(label_map.values()).index(label.item())]})"
            )  # Get class name
            plt.grid(True)
            plt.show()
        else:
            print("Could not plot, loaded data is corrupted")

    except ImportError:
        print("Matplotlib is not installed. Skipping plotting.")

    # --- Now, demonstrate training *without* caching ---
    print("\n--- Training with cache_capacity=0 (no caching) ---")
    dataset_no_cache = SDSSDataset(
        file_list, label_map, cache_capacity=0, device=str(device)
    )  # No cache
    dataloader_no_cache = DataLoader(
        dataset_no_cache,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Re-initialize the model (so we start from scratch)
    model = SimpleCNN(input_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(
            tqdm(dataloader_no_cache, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            if inputs.numel() == 0:
                continue

            # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader_no_cache):.4f}")
    print("Training finished (no cache)!")


if __name__ == "__main__":
    main()
