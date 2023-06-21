from yacs.config import CfgNode as CN
# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
DATASETS = CN()
DATASETS.TRAIN = ("structured3D_train",)
DATASETS.VAL   = ("structured3D_val",)
DATASETS.TEST  = ("structured3D_test",)
DATASETS.IMAGE = CN()
DATASETS.IMAGE.HEIGHT = 512
DATASETS.IMAGE.WIDTH  = 512

DATASETS.IMAGE.PIXEL_MEAN = [109.730, 103.832, 98.681]
DATASETS.IMAGE.PIXEL_STD  = [22.275, 22.124, 23.229]
DATASETS.IMAGE.TO_255 = True

DATASETS.PLANE_PARAMETERS = CN()
DATASETS.PLANE_PARAMETERS.MEAN = [0.0, 0.0, 0.0, 0.0]
DATASETS.PLANE_PARAMETERS.STD = [1.0, 1.0, 1.0, 1.0]

DATASETS.TARGET = CN()
DATASETS.TARGET.HEIGHT= 128
DATASETS.TARGET.WIDTH = 128
DATASETS.DISTANCE_TH = 0.02
DATASETS.NUM_STATIC_POSITIVE_LINES = 300
DATASETS.NUM_STATIC_NEGATIVE_LINES = 40
DATASETS.HFLIP = False
DATASETS.VFLIP = False
DATASETS.LINE_CLASS_TYPE='single'
DATASETS.DISABLE_CLASSES=False

# Temporary DIR we move data to. For example during training on cluster
DATASETS.TMP_DIR = None
#
