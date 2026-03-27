import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 64, 7, 7)
        return self.deconv(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_ae(
    model: Autoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    max_batches: int | None,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        seen = 0
        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            images = images.to(device)
            optimizer.zero_grad()
            recon, _ = model(images)
            loss = F.mse_loss(recon, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        epoch_loss = running_loss / seen
        history.append(epoch_loss)
        print(f"[AE ] Epoch {epoch:03d}/{epochs:03d}  mse={epoch_loss:.6f}")

    return history


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


def train_vae(
    model: VariationalAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    beta: float,
    max_batches: int | None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    model.train()

    for epoch in range(1, epochs + 1):
        total_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        seen = 0
        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(images)
            loss, recon_value, kl_value = vae_loss(recon, images, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_sum += loss.item() * batch_size
            recon_sum += recon_value * batch_size
            kl_sum += kl_value * batch_size
            seen += batch_size

        metrics = {
            "total": total_sum / seen,
            "recon": recon_sum / seen,
            "kl": kl_sum / seen,
        }
        history.append(metrics)
        print(
            f"[VAE] Epoch {epoch:03d}/{epochs:03d}  "
            f"total={metrics['total']:.6f}  recon={metrics['recon']:.6f}  kl={metrics['kl']:.6f}"
        )

    return history


@torch.no_grad()
def collect_examples(
    ae_model: Autoencoder,
    vae_model: VariationalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, _ = next(iter(test_loader))
    images = images.to(device)
    ae_recon, _ = ae_model(images)
    vae_recon, _, _, _ = vae_model(images)
    return images.cpu(), ae_recon.cpu(), vae_recon.cpu()


@torch.no_grad()
def collect_latents(
    ae_model: Autoencoder,
    vae_model: VariationalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ae_latents = []
    vae_latents = []
    labels_all = []

    for batch_idx, (images, labels) in enumerate(test_loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        images = images.to(device)
        _, ae_z = ae_model(images)
        _, mu, _, _ = vae_model(images)
        ae_latents.append(ae_z.cpu())
        vae_latents.append(mu.cpu())
        labels_all.append(labels)

    return torch.cat(ae_latents), torch.cat(vae_latents), torch.cat(labels_all)


def plot_training_curves(
    ae_history: list[float],
    vae_history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    epochs = range(1, len(ae_history) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, ae_history, marker="o", label="AE MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x["total"] for x in vae_history], marker="o", label="VAE Total")
    plt.plot(epochs, [x["recon"] for x in vae_history], marker="x", label="Recon")
    plt.plot(epochs, [x["kl"] for x in vae_history], marker="s", label="KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Variational Autoencoder")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def plot_reconstructions(
    images: torch.Tensor,
    ae_recon: torch.Tensor,
    vae_recon: torch.Tensor,
    output_dir: Path,
    samples: int,
) -> None:
    fig, axes = plt.subplots(3, samples, figsize=(1.8 * samples, 5))
    rows = [images, ae_recon, vae_recon]
    labels = ["Original", "AE", "VAE"]

    for row_idx, (row_images, label) in enumerate(zip(rows, labels)):
        for col_idx in range(samples):
            axes[row_idx, col_idx].imshow(row_images[col_idx].squeeze(), cmap="gray")
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=11)

    plt.suptitle("Reconstruction Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructions.png", dpi=160)
    plt.close()


def plot_latent_spaces(
    ae_latents: torch.Tensor,
    vae_latents: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
) -> None:
    if ae_latents.size(1) < 2 or vae_latents.size(1) < 2:
        print("Skip latent space plot because latent_dim < 2.")
        return

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(ae_latents[:, 0], ae_latents[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.title("AE Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")

    plt.subplot(1, 2, 2)
    plt.scatter(vae_latents[:, 0], vae_latents[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.title("VAE Latent Space (mu)")
    plt.xlabel("z1")
    plt.ylabel("z2")

    plt.tight_layout()
    plt.savefig(output_dir / "latent_spaces.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MNIST Autoencoder and VAE.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ae_vae_compare"))
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Outputs will be saved to: {output_dir}")

    train_loader, test_loader = build_dataloaders(args.batch_size)

    ae_model = Autoencoder(latent_dim=args.latent_dim).to(device)
    vae_model = VariationalAutoencoder(latent_dim=args.latent_dim).to(device)

    ae_history = train_ae(
        ae_model, train_loader, device, args.epochs, args.lr, args.max_train_batches
    )
    vae_history = train_vae(
        vae_model,
        train_loader,
        device,
        args.epochs,
        args.lr,
        args.beta,
        args.max_train_batches,
    )

    images, ae_recon, vae_recon = collect_examples(ae_model, vae_model, test_loader, device)
    ae_latents, vae_latents, labels = collect_latents(
        ae_model, vae_model, test_loader, device, args.max_test_batches
    )

    plot_training_curves(ae_history, vae_history, output_dir)
    plot_reconstructions(images, ae_recon, vae_recon, output_dir, args.samples)
    plot_latent_spaces(ae_latents, vae_latents, labels, output_dir)

    print("\nFinal metrics")
    print(f"AE final MSE       : {ae_history[-1]:.6f}")
    print(f"VAE final total    : {vae_history[-1]['total']:.6f}")
    print(f"VAE final recon    : {vae_history[-1]['recon']:.6f}")
    print(f"VAE final KL       : {vae_history[-1]['kl']:.6f}")


if __name__ == "__main__":
    main()
