from pathlib import Path
import tarfile
import urllib.request

# Official CIFAR-10 Python version
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Find the project root folder
project_root = Path(__file__).resolve().parent.parent

# Build all required paths
dataset_folder = project_root / "datasets"
archive_path = dataset_folder / "cifar-10-python.tar.gz"
extracted_folder = dataset_folder / "cifar-10-batches-py"

# Create the datasets folder if needed
dataset_folder.mkdir(exist_ok=True)

# Stop here if the dataset is already extracted
if extracted_folder.exists():
    print("CIFAR-10 is already available at:", extracted_folder)

else:
    # Download the archive only if it does not exist
    if archive_path.exists():
        print("Archive already exists:", archive_path)
    else:
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print("Download complete.")

    # Extract the archive
    print("Extracting CIFAR-10...")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=dataset_folder, filter="data")

    print("CIFAR-10 is ready at:", extracted_folder)