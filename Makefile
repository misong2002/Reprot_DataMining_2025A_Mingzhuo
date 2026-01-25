# ----------------------------
# 配置变量
# ----------------------------
DATA_DIR := data
URL := https://zenodo.org/records/17587229/files/dayabay_full_dataset_hdf5_1-0-0.zip?download=1
ZIP_FILE := $(DATA_DIR)/dayabay_full_dataset_hdf5_1-0-0.zip
UNZIP_DIR := $(DATA_DIR)/dayabay_full_dataset
HDF5_FILE := $(DATA_DIR)/hdf5/dayabay_dataset/dayabay_events_AD11.hdf5
PHYSICS_FILE := $(DATA_DIR)/physics_data.npz
LATENT_FILE := $(DATA_DIR)/latent_repr.npz
CLUSTERS_FILE := $(DATA_DIR)/clusters.npz
MATCHING_FILE := $(DATA_DIR)/matching.npz

.PHONY: all download unzip clean

# ----------------------------
# 默认目标
# ----------------------------
all: $(MATCHING_FILE)


# ----------------------------
# 下载
# ----------------------------
$(ZIP_FILE): | $(DATA_DIR)
	curl -L $(URL) -o $(ZIP_FILE)

download: $(ZIP_FILE)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# ----------------------------
# 解压
# ----------------------------
unzip: $(UNZIP_DIR)

$(UNZIP_DIR): $(ZIP_FILE)
	unzip -q $(ZIP_FILE) -d $(DATA_DIR)

# ----------------------------
# 数据处理
# ----------------------------
$(PHYSICS_FILE): $(UNZIP_DIR)
	python3 script/physics.py -d $(HDF5_FILE) -o $(PHYSICS_FILE)

$(LATENT_FILE): $(PHYSICS_FILE) 
	python3 script/dim_reducing.py --i $(PHYSICS_FILE) --o $(LATENT_FILE)

$(CLUSTERS_FILE): $(LATENT_FILE) 
	python3 script/cluster.py --input $(LATENT_FILE) --output $(CLUSTERS_FILE)


# ----------------------------
# 清理
# ----------------------------
clean:
	rm -rf $(ZIP_FILE) $(UNZIP_DIR) \
	       $(PHYSICS_FILE) $(LATENT_FILE) \
	       $(CLUSTERS_FILE) $(MATCHING_FILE)
