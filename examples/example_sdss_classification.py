import torch
import torchfits
import requests
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from astropy.io import fits
from tqdm import tqdm # for progress bar

# --- Data Download and Caching ---

def download_sdss_spectrum(plate, mjd, fiberid, base_url, save_dir):
    """Downloads an SDSS spectrum FITS file."""
    filename = f"spec-{plate:04}-{mjd}-{fiberid:04}.fits"
    filepath = os.path.join(save_dir, filename)
    url = f"{base_url}/{plate:04}/{filename}"

    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File already exists: {filepath}")
    return filepath

# --- PyTorch Dataset ---

class SDSSDataset(Dataset):
    def __init__(self, file_list, label_map):
        self.file_list = file_list
        self.label_map = label_map  # Dictionary mapping class strings to integer labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        try:
            # Read the spectrum data (flux) and the class (from header or table).
            # Assuming the flux is in HDU 1 and class is in HDU 2, 'CLASS' keyword.
            data = torchfits.read(filename, hdu=2) # Read HDU 2 (Binary Table)
            flux = data['flux'] # A torch Tensor
            
            # You could get the header using torchfits, and extract the CLASS keyword
            #header = torchfits.get_header(filename, 1)
            #obj_class = header.get('CLASS')
            # Or using fitsio to read that particular keyword, as an example:
            obj_class = fits.getval(filename, 'CLASS', ext=1)


            label = self.label_map[obj_class.strip()]  # Convert string to integer label

            # Normalize the flux (simple example)
            flux = flux / torch.max(flux)

            return flux, torch.tensor(label, dtype=torch.long) #Return also as tensor


        except Exception as e:
            print(f"Error reading or processing {filename}: {e}")
            # Return None if there's an error.  The collate_fn will handle it.
            return None

    def get_wavelength(self, idx):
        """
        Get wavelengths, for plotting proposes
        """
        filename = self.file_list[idx]
        data = torchfits.read(filename, hdu=2)
        return data['loglam']


# --- Model (Simple 1D CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2) #in_channels = 1
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
    return torch.utils.data.dataloader.default_collate(batch)


# --- Main Script ---

def main():
    base_url = "https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/v5_13_2/spectra/lite"  # Example URL
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Example plate/mjd/fiber combinations (replace with a larger sample if desired)
    spectra_to_download = [
        (266, 51630, 3),   # QSO
        (266, 51630, 4),  # GALAXY
        (266, 51630, 6), # STAR
        (334, 51689, 5) #STAR
    ]

    # Download files
    file_list = []
    for plate, mjd, fiberid in spectra_to_download:
        file_path = download_sdss_spectrum(plate, mjd, fiberid, base_url, data_dir)
        file_list.append(file_path)


    # Define class labels
    label_map = {"STAR": 0, "GALAXY": 1, "QSO": 2}
    num_classes = len(label_map)

    # Create Dataset and DataLoader
    dataset = SDSSDataset(file_list, label_map)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    #Get wavelengths, assuming they are the same
    wavelengths = np.power(10, dataset.get_wavelength(0).numpy())


    # Determine input size from the first spectrum
    # We need to check for None in case the first item failed to load.
    first_item = dataset[0]
    if first_item is None:
        raise RuntimeError("First item in dataset is None. Check for file download issues.")
    input_size = first_item[0].shape[0]


    # Initialize model, loss, and optimizer
    model = SimpleCNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

     # --- Training Loop (Simplified) ---
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Skip if the batch is empty (due to None values)
            if inputs.numel() == 0:
                continue
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # Get predicted class
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
          plt.title(f"SDSS Spectrum (Class: {list(label_map.keys())[list(label_map.values()).index(label.item())]})") # Get class name
          plt.grid(True)
          plt.show()
        else:
          print("Could not plot, loaded data is corrupted")

    except ImportError:
        print("Matplotlib is not installed. Skipping plotting.")


if __name__ == "__main__":
    main()
    