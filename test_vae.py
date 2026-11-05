import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from vae_model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Load trained model
model = VAE(latent_dim=20).to(device)
model.load_state_dict(
    torch.load("fashion_vae.pth", map_location=device)
)
model.eval()

# Select 5 test images
images = torch.stack([dataset[i][0] for i in range(5)]).to(device)

# Reconstruct images
with torch.no_grad():
    reconstructed, _, _ = model(images)

# Display original and reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(5):
    axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstructed[i].cpu().squeeze(), cmap="gray"
    )
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("vae_reconstruction.png")
plt.show()

print("Reconstruction test completed!")