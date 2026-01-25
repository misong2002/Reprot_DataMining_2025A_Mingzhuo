#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.optimize import minimize

# ============================================================
# Background spectra loader
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


# ============================================================
# MLE with fixed IBD template
# ============================================================
def fit_mle_with_ibd_template(
    cluster_spectrum,
    background_spectrum_dict,
    source_list,
    ibd_template,
):
    C, B = cluster_spectrum.shape
    S = len(source_list)

    # -------- templates --------
    T_bkg = np.zeros((S, B))
    N_prior = np.zeros(S)
    sigma_prior = np.zeros(S)

    for i, s in enumerate(source_list):
        count, unc, spec = background_spectrum_dict[s]
        T_bkg[i] = spec / spec.sum()
        N_prior[i] = count
        sigma_prior[i] = max(unc, 1e-6)

    T_ibd = ibd_template

    # -------- parameters --------
    # x = [ f(c,s) , f(c,IBD) ]
    n_fb = C * S
    n_fi = C

    def unpack(x):
        f_bkg = x[:n_fb].reshape(C, S)
        f_ibd = x[n_fb:]
        return f_bkg, f_ibd

    def nll(x):
        f_bkg, f_ibd = unpack(x)

        mu = np.einsum('cs,sb->cb', f_bkg, T_bkg)
        mu += f_ibd[:, None] * T_ibd[None, :]
        mu = np.clip(mu, 1e-9, None)

        # Poisson likelihood
        nll_poi = np.sum(mu - cluster_spectrum * np.log(mu))

        # Gaussian prior on total background counts
        f_sum = f_bkg.sum(axis=0)
        nll_gauss = 0.5 * np.sum(((f_sum - N_prior) / sigma_prior) ** 2)

        return nll_poi + nll_gauss

    # -------- initial values --------
    f_bkg0 = np.zeros((C, S))
    for s in range(S):
        f_bkg0[:, s] = N_prior[s] / C

    f_ibd0 = np.maximum(
        cluster_spectrum.sum(axis=1) - f_bkg0.sum(axis=1),
        1.0,
    )

    x0 = np.concatenate([f_bkg0.ravel(), f_ibd0])
    bounds = [(0, None)] * len(x0)

    res = minimize(
        nll,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
    )

    if not res.success:
        raise RuntimeError(res.message)

    return unpack(res.x), N_prior


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters', required=True)
    parser.add_argument('--physics', required=True)
    parser.add_argument('--period', default='8AD', choices=['6AD', '7AD', '8AD'])
    parser.add_argument('--detector', default='AD11')
    parser.add_argument('--output', default='data/matching.npz')
    args = parser.parse_args()

    labels = np.load(args.clusters)['labels']
    energy_prompt = np.load(args.physics)['energy_prompt_MeV']

    bkg_dict, energy_list, source_list = load_background_spectra(
        args.period, args.detector
    )
    if energy_list is None:
        raise RuntimeError("Energy grids not consistent")

    ibd_energy, ibd_spec = load_ibd_spectrum(args.period, args.detector)
    if not np.array_equal(ibd_energy, energy_list):
        raise RuntimeError("IBD spectrum energy grid mismatch")

    cluster_spec, cluster_ids = generate_cluster_spectrum(
        energy_list, labels, energy_prompt
    )

    (f_bkg, f_ibd), N_true = fit_mle_with_ibd_template(
        cluster_spec,
        bkg_dict,
        source_list,
        ibd_spec,
    )

    fitted_bkg = f_bkg.sum(axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        f_background=f_bkg,
        f_ibd=f_ibd,
        cluster_ids=cluster_ids,
        energy_MeV=energy_list,
        source_list=np.array(source_list),
        fitted_background_counts=fitted_bkg,
        true_background_counts=N_true,
    )

    print("=== Background count comparison ===")
    for s, f, t in zip(source_list, fitted_bkg, N_true):
        print(f"{s:20s}  fitted = {f:10.1f}   truth = {t:10.1f}")


if __name__ == "__main__":
    main()
