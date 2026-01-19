import argparse
from random import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(21)

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(
    description="Learn low-dimensional representation of physics events using an autoencoder"
)
parser.add_argument(
    "--i",
    type=str,
    default="data/physics_data.npz",
    help="Path to input .npz file containing structured array (default: data/physics_data.npz)",
)
parser.add_argument(
    "--o",
    type=str,
    default="data/latent_repr.npz",
    help="Path to output .npz file for latent representation (default: data/latent_repr.npz)",
)
parser.add_argument(
    "--ldim",
    type=int,
    default=2,
    help="Dimensionality of latent space (default: 2)",
)

args = parser.parse_args()

DATA_PATH = args.i
OUTPUT_PATH = args.o
LATENT_DIM = args.ldim

# ----------------------------
# Training configuration
# ----------------------------
BATCH_SIZE = 1024
EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# ----------------------------
# Load data
# ----------------------------
npz = np.load(DATA_PATH)
data = npz[npz.files[0]]  # structured array

feature_names = [
    "energy_prompt_MeV",
    "energy_delayed_MeV",
    "log_time_diff_us",
    "log_vertex_distance_mm",
]

X = np.vstack([data[name] for name in feature_names]).T.astype(np.float32)

# ----------------------------
# Standardize features
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Dataset / DataLoader
# ----------------------------
tensor_X = torch.tensor(X_scaled)
dataset = TensorDataset(tensor_X)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Autoencoder model
# ----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
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
            nn.Linear(16, 4),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


model = AutoEncoder(LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()



# ----------------------------
# Training loop
# ----------------------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        x_hat, _ = model(batch)
        loss = criterion(x_hat, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / len(dataset)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | MSE loss = {avg_loss:.6f}")

# ----------------------------
# Extract latent representation
# ----------------------------
model.eval()
with torch.no_grad():
    _, Z = model(tensor_X.to(DEVICE))
    Z = Z.cpu().numpy()

# ----------------------------
# Save result
# ----------------------------
np.savez(
    OUTPUT_PATH,
    latent=Z,
    original_features=X,
    scaled_features=X_scaled,
    feature_names=feature_names,
    latent_dim=LATENT_DIM,
)

torch.save({
    "encoder_state": model.encoder.state_dict(),
    "decoder_state": model.decoder.state_dict(),
    "x_mean": scaler.mean_,
    "x_std": scaler.scale_,
}, "data/autoencoder_checkpoint.pt")


print(f"Latent representation saved to {OUTPUT_PATH}")
