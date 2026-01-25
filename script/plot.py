import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import h5py


# ============================================================
# Physics histograms
# ============================================================
def plot_physics_histograms(data_physics, quantity_keys, outpath, bins=100):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for ax, key in zip(axs, quantity_keys):
        ax.hist(data_physics[key], bins=bins, alpha=0.7)
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Counts")

    for ax in axs[len(quantity_keys):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================
# Latent space
# ============================================================
def plot_latent_space(latent, outpath, bins=1000):
    plt.figure(figsize=(8, 6))
    plt.hist2d(latent[:, 0], latent[:, 1], bins=bins, norm=LogNorm())
    plt.colorbar(label="Counts")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space distribution")
    plt.savefig(outpath)
    plt.close()


def plot_clusters_latent(
    latent, labels, outdir,
    n_clusters=14,
    bins_local=500,
    bins_global=(np.linspace(-2.5, 10, 1000),
                 np.linspace(-2.5, 10, 1000))
):
    z1, z2 = latent[:, 0], latent[:, 1]

    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue

        with PdfPages(outdir / f"cluster_{cid:02d}_latent.pdf") as pdf:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            _, xedges, yedges, _ = axs[0].hist2d(
                z1[mask], z2[mask],
                bins=bins_local,
                norm=LogNorm()
            )
            axs[0].set_title(f"Cluster {cid} (local bins)")

            axs[1].hist2d(
                z1[mask], z2[mask],
                bins=bins_global,
                norm=LogNorm()
            )
            axs[1].plot(xedges, np.full_like(xedges, yedges[0]), "r")
            axs[1].plot(xedges, np.full_like(xedges, yedges[-1]), "r")
            axs[1].plot(np.full_like(yedges, xedges[0]), yedges, "r")
            axs[1].plot(np.full_like(yedges, xedges[-1]), yedges, "r")
            axs[1].set_title(f"Cluster {cid} (global bins)")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_cluster_histograms(data_physics, labels, cluster_ids, outpath, bins=300):
    keys = list(data_physics.keys())[:4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for ax, key in zip(axs, keys):

        # --------------------------------------
        # Handle log_ quantities
        # --------------------------------------
        if key.startswith("log_"):
            plot_key = key[4:]
            stacks = [
                np.exp(data_physics[key][labels == cid])
                for cid in cluster_ids
            ]
            ax.set_xlabel(plot_key)
        else:
            plot_key = key
            stacks = [
                data_physics[key][labels == cid]
                for cid in cluster_ids
            ]
            ax.set_xlabel(plot_key)

        ax.hist(
            stacks,
            bins=bins,
            stacked=True,
            label=[f"C{c}" for c in cluster_ids]
        )

        ax.set_title(plot_key)
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()



# ============================================================
# Background & cluster spectra
# ============================================================
def load_background_spectra(period_name, detector_name):
    source_list = [
        'accidentals',
        'lithium_helium',
        'fast_neutrons',
        'amc',
        'alpha_neutron',
    ]

    data_path = f'data/hdf5/dayabay_dataset/dayabay_events_{detector_name}.hdf5'
    with h5py.File(data_path, 'r') as f:
        events = f['events'][:]

    events = events[np.abs(events['n_det'] - int(period_name[0])) < 0.1]
    time_s = events['time_prompt_s']
    livetime_days = (time_s.max() - time_s.min()) / (24 * 3600)

    rate_path = 'data/hdf5/dayabay_dataset/dayabay_background_rates.hdf5'
    with h5py.File(rate_path, 'r') as f:
        rates = f[period_name][:]

    count_dict = {}
    for r in rates:
        key = str(r[0])[2:-1]
        count_dict[key] = r[1] * livetime_days

    spec_path = f'data/hdf5/dayabay_dataset/dayabay_background_spectra_{period_name}.hdf5'
    background_spectrum_dict = {}
    energy_list = []

    with h5py.File(spec_path, 'r') as f:
        for s in source_list:
            data = f[f'spectrum_shape_{s}_{detector_name}'][:]
            energy = data['E_min_MeV']
            spectrum = data['N'].astype(float)

            count = count_dict[f'{s}_rate']
            unc = count_dict[f'{s}_uncertainty']

            background_spectrum_dict[s] = (count, unc, energy, spectrum)
            energy_list.append(energy)

    common_energy = None
    if all(np.array_equal(energy_list[0], e) for e in energy_list[1:]):
        common_energy = energy_list[0]
        for s in source_list:
            c, u, _, spec = background_spectrum_dict[s]
            background_spectrum_dict[s] = (c, u, spec)

    return background_spectrum_dict, common_energy, source_list




# ============================================================
# Cluster spectrum
# ============================================================
def generate_cluster_spectrum(energy_list, labels, energy_prompt_MeV):
    unique_labels = np.unique(labels)
    n_cluster = unique_labels.size
    n_bin = energy_list.size

    bin_edges = np.concatenate([energy_list, [np.inf]])
    cluster_spectrum = np.zeros((n_cluster, n_bin))

    for i, lb in enumerate(unique_labels):
        mask = labels == lb
        hist, _ = np.histogram(energy_prompt_MeV[mask], bins=bin_edges)
        cluster_spectrum[i] = hist

    return cluster_spectrum, unique_labels


def plot_cluster10_vs_lithium_helium(
    background_spectrum_dict,
    common_energy,
    cluster_spectrum,
    unique_labels,
    output_path
):
    """
    Compare cluster 10 energy spectrum with lithium-helium background,
    including uncertainty band.
    """

    # --------------------------------------------------
    # Locate cluster 10
    # --------------------------------------------------
    if 10 not in unique_labels:
        raise ValueError("Cluster 10 not found in unique_labels.")

    cluster_index = np.where(unique_labels == 10)[0][0]
    cluster_hist = cluster_spectrum[cluster_index]

    # --------------------------------------------------
    # Lithium-Helium background
    # --------------------------------------------------
    count, unc, shape = background_spectrum_dict["lithium_helium"]

    expected = shape * count
    lower = shape * (count - unc)
    upper = shape * (count + unc)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.step(
        common_energy,
        cluster_hist,
        where="post",
        linewidth=2,
        label="Cluster 10"
    )

    plt.plot(
        common_energy,
        expected,
        color="tab:orange",
        linewidth=2,
        label="Lithium–Helium"
    )

    plt.fill_between(
        common_energy,
        lower,
        upper,
        color="tab:orange",
        alpha=0.3,
        label="Lithium–Helium uncertainty"
    )

    plt.xlabel("Prompt Energy (MeV)")
    plt.ylabel("Counts")
    plt.title("Cluster 10 vs Lithium–Helium Background")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics", required=True)
    parser.add_argument("--latent", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--detector", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    physics = np.load(args.physics)
    latent = np.load(args.latent)["latent"]
    labels = np.load(args.clusters)["labels"]

    # 1. physics histograms
    plot_physics_histograms(
        physics,
        list(physics.keys())[:4],
        outdir / "physics_hist.pdf"
    )

    # 2. latent space
    plot_latent_space(latent, outdir / "latent_2d_hist.pdf")

    # 3. per-cluster latent
    plot_clusters_latent(latent, labels, outdir)

    # 4. grouped clusters
    plot_cluster_histograms(
        physics, labels,
        list(range(8)),
        outdir / "clusters_0_7.pdf"
    )
    plot_cluster_histograms(
        physics, labels,
        [9, 11],
        outdir / "clusters_9_11.pdf"
    )
    plot_cluster_histograms(
        physics, labels,
        [8, 10, 12],
        outdir / "clusters_8_10_12.pdf"
    )

    # 5. lithium vs cluster 10
    bkg_dict, common_energy, _ = load_background_spectra(args.period, args.detector)

    cluster_spectrum, unique_labels = generate_cluster_spectrum(
        common_energy,
        labels,
        physics["energy_prompt_MeV"]
    )

    plot_cluster10_vs_lithium_helium(
        bkg_dict,
        common_energy,
        cluster_spectrum,
        unique_labels,
        outdir / "cluster10_vs_lithium_helium.pdf"
    )


if __name__ == "__main__":
    main()
