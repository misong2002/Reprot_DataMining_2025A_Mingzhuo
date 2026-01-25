# Data-Driven Clustering Analysis of Daya Bay Events

This repository contains an exploratory data mining analysis of publicly available data from the **Daya Bay reactor antineutrino experiment**. The goal is to study whether unsupervised clustering methods can separate inverse beta decay (IBD) events from different background populations based purely on event-level observables, without imposing explicit physics-driven selection cuts.

---

## Overview

Electron antineutrinos in Daya Bay are detected via the inverse beta decay (IBD) process

[ \bar{\nu}_e + p \rightarrow e^+ + n ]

which produces a characteristic prompt--delayed coincidence signature. In addition to genuine IBD events, the dataset contains several background components, including accidental coincidences and correlated cosmogenic backgrounds such as (^{9}\mathrm{Li}/^{8}\mathrm{He}).

In this project:

* Four physically motivated observables are used to describe each event:

  * prompt energy,
  * delayed energy,
  * spatial distance between prompt and delayed vertices,
  * time difference between prompt and delayed signals.
* Events are embedded into a **two-dimensional latent space** using an encoder--decoder architecture.
* A **hierarchical clustering strategy** (HDBSCAN (\rightarrow) DBSCAN (\rightarrow) K-means) is applied in the latent space to handle strong class imbalance.
* The resulting clusters are interpreted by examining their distributions in the original physical observables.

This study serves as a data-driven complement to traditional physics-based analyses in reactor neutrino experiments.

---

## Dataset

The analysis uses the official Daya Bay public dataset hosted on Zenodo:

* **Dataset**: `dayabay_full_dataset_hdf5_1-0-0`
* **Source**: Zenodo (Daya Bay Collaboration)

Only data from the **8AD operation period** are used, and the analysis shown in the report focuses on **detector AD11**.

---

## Analysis Pipeline

1. **Feature extraction**
   Event-level physical observables are extracted from the HDF5 dataset.

2. **Representation learning**
   The four-dimensional feature space is mapped into a two-dimensional latent space using an encoder--decoder model.

3. **Hierarchical clustering**

   * **HDBSCAN** identifies dense, core clusters (IBD-dominated regions).
   * **DBSCAN** further separates near and far structures among lower-density events.
   * **K-means** subdivides remaining background-like events.

4. **Physics interpretation**
   Cluster labels are projected back onto the original observables, and histograms of prompt energy, delayed energy, spatial separation, and time difference are analyzed.

---

## Dependencies

The core Python dependencies required to run the analysis scripts and plotting utilities are:

* Python (\geq 3.9)
* `numpy`
* `scipy`
* `h5py`
* `matplotlib`
* `scikit-learn`
* `hdbscan`

For convenience, dependencies can be installed via:

```bash
pip install numpy scipy h5py matplotlib scikit-learn hdbscan
```

If representation learning is performed from scratch, an additional deep learning framework (e.g. `pytorch` or `tensorflow`) is required, depending on the encoder--decoder implementation.

---

## Usage

### 1. Data preparation

Download and unzip the Daya Bay HDF5 dataset into a local directory, for example:

```
make download
```

### 2. Feature extraction

Run the feature extraction script to select the desired operation period, detector, and observables. The output is typically stored as NumPy arrays or HDF5 files for downstream analysis.
```bash
make data/latent_repr.npz
```

### 3. Latent space clustering

Cluster the latent representations using the hierarchical pipeline. A typical workflow is:

```bash
make data/cluster.npz
```

(Exact arguments depend on the specific clustering script.)

### 4. Physics histogram visualization

Clustered events can be visualized using grouped histograms of physical observables. For observables stored in logarithmic form (prefixed with `log_`), values are automatically exponentiated before plotting.

Example usage:

```
make plots
```

---

## Results Summary

* High-density clusters in the latent space show prompt--delayed distance and time difference distributions consistent with genuine IBD events.
* Several low-density, elongated clusters exhibit flat or uncorrelated distributions and are interpreted as **accidental coincidence–dominated** samples.
* A small cluster shows a prompt energy spectrum compatible with expectations for **Li--He cosmogenic backgrounds** in the low-energy region.

Not all background components can be cleanly separated due to strong experimental background suppression in the public dataset. The clustering algorithm preferentially isolates the most distinctive background events.

---

## Outlook

With access to datasets containing less aggressive background suppression, this data-driven pipeline could be further extended to:

* improve background characterization,
* assist in particle identification,
* complement traditional cut-based and likelihood-based analyses in neutrino experiments.

---

## Reference

F. P. An *et al.* (Daya Bay Collaboration),
*Precision Measurement of Reactor Antineutrino Oscillation at Kilometer-Scale Baselines by Daya Bay*,
**Phys. Rev. Lett. 130**, 161802 (2023).
