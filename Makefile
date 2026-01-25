# ----------------------------
# User-configurable parameters
# (can be overridden from command line)
# ----------------------------
PERIOD   ?= 8AD
DETECTOR ?= AD11

# ----------------------------
# Directory configuration
# ----------------------------
DATA_DIR   := data
SCRIPT_DIR := script
PLOT_SCRIPT := $(SCRIPT_DIR)/make_plots.py

# ----------------------------
# Dataset download configuration
# ----------------------------
URL := https://zenodo.org/records/17587229/files/dayabay_full_dataset_hdf5_1-0-0.zip?download=1
ZIP_FILE := $(DATA_DIR)/dayabay_full_dataset_hdf5_1-0-0.zip
UNZIP_DIR := $(DATA_DIR)/dayabay_full_dataset

# ----------------------------
# Input / output data files
# ----------------------------
HDF5_DIR  := $(DATA_DIR)/hdf5/dayabay_dataset
HDF5_FILE := $(HDF5_DIR)/dayabay_events_$(DETECTOR).hdf5

PHYSICS_FILE  := $(DATA_DIR)/physics_data.npz
LATENT_FILE   := $(DATA_DIR)/latent_repr.npz
CLUSTERS_FILE := $(DATA_DIR)/clusters.npz

# Final plot output directory
PLOT_DIR := $(DATA_DIR)/plots

.PHONY: all download unzip plots clean

# ----------------------------
# Default target:
# run the full pipeline and produce plots
# ----------------------------
all: plots

# ----------------------------
# Download raw dataset
# ----------------------------
download: $(ZIP_FILE)

$(ZIP_FILE): | $(DATA_DIR)
	curl -L $(URL) -o $(ZIP_FILE)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# ----------------------------
# Unzip dataset
# ----------------------------
unzip: $(UNZIP_DIR)

$(UNZIP_DIR): $(ZIP_FILE)
	unzip -q $(ZIP_FILE) -d $(DATA_DIR)

# ----------------------------
# Data processing pipeline
# ----------------------------
# 1. Extract physical quantities from HDF5
$(PHYSICS_FILE): $(UNZIP_DIR)
	python3 $(SCRIPT_DIR)/physics.py \
	    -d $(HDF5_FILE) \
	    -o $(PHYSICS_FILE)

# 2. Dimensionality reduction (latent representation)
$(LATENT_FILE): $(PHYSICS_FILE)
	python3 $(SCRIPT_DIR)/dim_reducing.py \
	    --i $(PHYSICS_FILE) \
	    --o $(LATENT_FILE)

# 3. Clustering in latent space
$(CLUSTERS_FILE): $(LATENT_FILE)
	python3 $(SCRIPT_DIR)/cluster.py \
	    --input  $(LATENT_FILE) \
	    --output $(CLUSTERS_FILE)

# ----------------------------
# Final plotting step
# This is the main analysis product
# ----------------------------
plots: $(PHYSICS_FILE) $(LATENT_FILE) $(CLUSTERS_FILE)
	mkdir -p $(PLOT_DIR)
	python3 $(PLOT_SCRIPT) \
	    --physics  $(PHYSICS_FILE) \
	    --latent   $(LATENT_FILE) \
	    --clusters $(CLUSTERS_FILE) \
	    --outdir   $(PLOT_DIR) \
	    --period   $(PERIOD) \
	    --detector $(DETECTOR)

# ----------------------------
# Cleanup generated files
# ----------------------------
clean:
	rm -rf $(ZIP_FILE) $(UNZIP_DIR) \
	       $(PHYSICS_FILE) $(LATENT_FILE) \
	       $(CLUSTERS_FILE) $(PLOT_DIR)
