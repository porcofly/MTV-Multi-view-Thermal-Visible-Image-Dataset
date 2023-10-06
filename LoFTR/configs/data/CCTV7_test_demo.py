from configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/CCTV7_test_5.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MGDPT_DF = 8
cfg.DATASET.MGDPT_IMG_PAD = False
cfg.DATASET.MGDPT_DEPTH_PAD = False
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
