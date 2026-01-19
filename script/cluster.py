import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def main(args):
    # ---------- load latent ----------
    data = np.load(args.input)
    z = data["latent"]  # shape (N, latent_dim)

    # ---------- optional re-standardization ----------
    # GMM works better if axes are comparable
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # ---------- fit GMM ----------
    gmm = GaussianMixture(
        n_components=args.Nclusters,
        covariance_type="full",
        n_init=50,
        random_state=42,
    )

    labels = gmm.fit_predict(z_scaled)
    probs = gmm.predict_proba(z_scaled)

    # ---------- save results ----------
    np.savez(
        args.output,
        latent=z,
        latent_scaled=z_scaled,
        labels=labels,
        probabilities=probs,
        means=gmm.means_,
        covariances=gmm.covariances_,
        weights=gmm.weights_,
    )

    print(f"GMM clustering finished.")
    print(f"Saved to: {args.output}")
    print(f"Clusters: {args.Nclusters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMM clustering on autoencoder latent space"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/latent.npz",
        help="Input npz file containing latent array (key: latent)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gmm_clusters.npz",
        help="Output npz file for clustering results",
    )
    parser.add_argument(
        "--Nclusters",
        type=int,
        default=3,
        help="Number of GMM components",
    )

    args = parser.parse_args()
    main(args)
