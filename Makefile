# 定义变量
DATA_DIR := data
URL := https://zenodo.org/records/17587229/files/dayabay_full_dataset_hdf5_1-0-0.zip?download=1
OUT_FILE := $(DATA_DIR)/dayabay_full_dataset_hdf5_1-0-0.zip

.PHONY: download

# 默认目标：下载文件
all: $(OUT_FILE)

# 目标：data 目录
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# 下载文件（用 curl）
$(OUT_FILE): | $(DATA_DIR)
	curl -L $(URL) -o $(OUT_FILE)

# 如果你想用 wget 替代 curl，可以把上面那条替换成：
#   wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -O $(OUT_FILE) "$(URL)"

download: $(OUT_FILE)

.PHONY: clean

clean:
	rm -rf $(OUT_FILE)

UNZIP_DIR := $(DATA_DIR)/dayabay_full_dataset

# 解压目标
unzip: $(UNZIP_DIR)

$(UNZIP_DIR): $(OUT_FILE)
	unzip -q $(OUT_FILE) -d $(DATA_DIR)
	touch $(UNZIP_DIR)

physics_data.npz: $(UNZIP_DIR)
	python3 script/physics.py -d $(DATA_DIR)/hdf5/dayabay_dataset/dayabay_events_AD11.hdf5 -o data/physics_data.npz

latent_repr.npz: physics_data.npz
	python3 script/dim_reducing.py --i data/physics_data.npz --o data/latent_repr.npz --ldim 3

gmm_clusters.npz: latent_repr.npz
	python3 script/cluster.py --input data/latent_repr.npz --output data/gmm_clusters.npz --Nclusters 4