import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae_model import VAE


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", device)


# Fashion-MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)


# Initialize VAE
model = VAE(latent_dim=20).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# VAE loss function
def vae_loss(reconstructed, original, mu, logvar):

    reconstruction_loss = F.binary_cross_entropy(
        reconstructed,
        original,
        reduction="sum"
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_divergence


# Training
epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        optimizer.zero_grad()

        reconstructed, mu, logvar = model(images)

        loss = vae_loss(
            reconstructed,
            images,
            mu,
            logvar
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataset)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# Save trained model
torch.save(
    model.state_dict(),
    "fashion_vae.pth"
)

print("\nVAE training completed!")
print("Model saved as fashion_vae.pth")
