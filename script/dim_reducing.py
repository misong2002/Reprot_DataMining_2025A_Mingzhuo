import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data loading & preprocessing
# ----------------------------
def load_and_scale_data(npz_path, feature_names):
    npz = np.load(npz_path)
    data = npz[npz.files[0]]  # structured array

    X = np.vstack([data[name] for name in feature_names]).T.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler


def make_dataloader(X_scaled, batch_size):
    tensor_X = torch.tensor(X_scaled)
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return tensor_X, dataset, loader


# ----------------------------
# Model
# ----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ----------------------------
# Training
# ----------------------------
def train_autoencoder(
    model, loader, dataset_size, device, epochs, lr
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / dataset_size
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | MSE loss = {avg_loss:.6f}")


# ----------------------------
# Latent extraction
# ----------------------------
def extract_latent(model, tensor_X, device):
    model.eval()
    with torch.no_grad():
        _, Z = model(tensor_X.to(device))
    return Z.cpu().numpy()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Learn low-dimensional representation of physics events using an autoencoder"
    )
    parser.add_argument("--i", type=str, default="data/physics_data.npz")
    parser.add_argument("--o", type=str, default="data/latent_repr.npz")
    parser.add_argument("--ldim", type=int, default=2)
    args = parser.parse_args()

    set_seed(21)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    feature_names = [
        "energy_prompt_MeV",
        "energy_delayed_MeV",
        "log_time_diff_us",
        "log_vertex_distance_mm",
    ]

    X, X_scaled, scaler = load_and_scale_data(args.i, feature_names)
    tensor_X, dataset, loader = make_dataloader(X_scaled, batch_size=1024)

    model = AutoEncoder(input_dim=4, latent_dim=args.ldim).to(DEVICE)

    train_autoencoder(
        model=model,
        loader=loader,
        dataset_size=len(dataset),
        device=DEVICE,
        epochs=200,
        lr=1e-3,
    )

    Z = extract_latent(model, tensor_X, DEVICE)

    Path(args.o).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.o,
        latent=Z,
        original_features=X,
        scaled_features=X_scaled,
        feature_names=feature_names,
        latent_dim=args.ldim,
    )

    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "decoder_state": model.decoder.state_dict(),
            "x_mean": scaler.mean_,
            "x_std": scaler.scale_,
        },
        "data/autoencoder_checkpoint.pt",
    )

    print(f"Latent representation saved to {args.o}")


if __name__ == "__main__":
    main()
