from configs.data.base import cfg

TEST_BASE_PATH = "/media/yan/data/CCTV7/Loftr"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "/media/yan/data/CCTV7/Loftr"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/test_list.txt"

cfg.DATASET.MGDPT_IMG_RESIZE =640
cfg.DATASET.MGDPT_DF = 8
cfg.DATASET.MGDPT_IMG_PAD = False
cfg.DATASET.MGDPT_DEPTH_PAD = False
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
