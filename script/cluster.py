import argparse
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import hdbscan


# ----------------------------
# Load latent space
# ----------------------------
def load_latent(npz_path):
    data = np.load(npz_path)
    if "latent" not in data:
        raise KeyError("Input npz must contain key 'latent'")
    return data["latent"]


# ----------------------------
# Standardization
# ----------------------------
def standardize_latent(z):
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    return z_scaled.astype(np.float32), scaler


# ----------------------------
# Run HDBSCAN (level 1)
# ----------------------------
def run_hdbscan(z, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(z)
    return clusterer, labels


# ----------------------------
# Run DBSCAN (level 2)
# ----------------------------
def run_dbscan(z, eps, min_samples):
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = db.fit_predict(z)
    return db, labels


# ----------------------------
# Run Single-linkage (level 3)
# ----------------------------
def run_single_linkage(z, n_clusters):
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="single",
        affinity="euclidean",
    )
    labels = agg.fit_predict(z)
    return labels


# ----------------------------
# Main
# ----------------------------
def main(args):
    print("Hierarchical clustering pipeline started")

    # ---- load & scale ----
    z = load_latent(args.input)
    z_scaled, scaler = standardize_latent(z)
    N = len(z_scaled)

    final_labels = -np.ones(N, dtype=int)
    current_label = 0

    # ======================================================
    # 1) HDBSCAN level 1 (coarse stable clusters)
    # ======================================================
    hdb1, labels1 = run_hdbscan(
        z_scaled,
        min_cluster_size=args.min_cluster_size_1,
        min_samples=args.min_samples_1,
    )

    print("HDBSCAN-1 clusters:", np.unique(labels1[labels1 >= 0]).size)

    for cid, persistence in enumerate(hdb1.cluster_persistence_):
        mask = labels1 == cid
        if persistence > args.persistence_1:
            final_labels[mask] = current_label
            current_label += 1
        else:
            final_labels[mask] = -1  # low-persistence clusters treated as noise

    mask_lvl1_noise = final_labels == -1
    z_lvl1_noise = z_scaled[mask_lvl1_noise]

    print("Assigned labels after HDBSCAN-1:", np.unique(final_labels[final_labels >= 0]).size)

    # ======================================================
    # 2) DBSCAN level 2 (on noise + low-persistence clusters)
    # ======================================================
    if len(z_lvl1_noise) > 0:
        db2, labels2 = run_dbscan(
            z_lvl1_noise,
            eps=args.dbscan_eps_2,
            min_samples=args.min_samples_2,
        )

        print("DBSCAN-2 clusters:", np.unique(labels2[labels2 >= 0]).size)
        print("Noise rate:", np.sum(labels2 == -1) / len(labels2))

        noise_indices_lvl1 = np.where(mask_lvl1_noise)[0]

        for cid in np.unique(labels2):
            if cid == -1:
                continue
            mask = labels2 == cid
            final_labels[noise_indices_lvl1[mask]] = current_label
            current_label += 1

    mask_lvl2_noise = final_labels == -1
    z_lvl2_noise = z_scaled[mask_lvl2_noise]

    print("Assigned labels after DBSCAN-2:", np.unique(final_labels[final_labels >= 0]).size)

    # ======================================================
    # 3) Single-linkage (forced covering of remaining points)
    # ======================================================
    if len(z_lvl2_noise) > 0:
        print(f"Single-linkage on {len(z_lvl2_noise)} remaining points")
        sl_labels = run_single_linkage(z_lvl2_noise, args.single_linkage_clusters)
        noise_indices_lvl2 = np.where(mask_lvl2_noise)[0]

        for k in range(args.single_linkage_clusters):
            final_labels[noise_indices_lvl2[sl_labels == k]] = current_label
            current_label += 1

    # ======================================================
    # Save
    # ======================================================
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        labels=final_labels,
        latent=z,
        latent_scaled=z_scaled,
        params=dict(
            min_cluster_size_1=args.min_cluster_size_1,
            min_samples_1=args.min_samples_1,
            persistence_1=args.persistence_1,
            dbscan_eps_2=args.dbscan_eps_2,
            min_samples_2=args.min_samples_2,
            single_linkage_clusters=args.single_linkage_clusters,
            seed=args.seed,
        ),
    )

    print("Clustering finished")
    print("Total clusters:", np.unique(final_labels[final_labels >= 0]).size)
    print("Saved to:", args.output)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering: HDBSCAN → DBSCAN → Single-linkage"
    )

    parser.add_argument("--input", type=str, default="data/latent.npz")
    parser.add_argument("--output", type=str, default="data/clusters.npz")

    # HDBSCAN level 1
    parser.add_argument("--min_cluster_size_1", type=int, default=400)
    parser.add_argument("--min_samples_1", type=int, default=50)
    parser.add_argument("--persistence_1", type=float, default=1e-2)

    # DBSCAN level 2
    parser.add_argument("--dbscan_eps_2", type=float, default=0.1)
    parser.add_argument("--min_samples_2", type=int, default=100)

    # Single-linkage
    parser.add_argument("--single_linkage_clusters", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
