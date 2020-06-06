"""Basic experiments configuration
For different tasks, a specific configuration might be created by importing this basic config.
"""

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Resume
# ---------------------------------------------------------------------------- #
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True
# Path of weights to resume
_C.RESUME_PATH = ''

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ''

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0
# Whether to drop last
_C.DATALOADER.DROP_LAST = True

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = ''

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.BASE_LR = 0.001
_C.OPTIMIZER.WEIGHT_DECAY = 0.0

# Specific parameters of optimizers
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.momentum = 0.9
_C.OPTIMIZER.SGD.dampening = 0.0

_C.OPTIMIZER.Adam = CN()
_C.OPTIMIZER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ''

_C.SCHEDULER.MAX_ITERATION = 1
# Minimum learning rate. 0.0 for disable.
_C.SCHEDULER.CLIP_LR = 0.0

# Specific parameters of schedulers
_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Batch size
_C.TRAIN.BATCH_SIZE = 1
# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = 0
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = 50
# Period to summary training status. 0 for disable
_C.TRAIN.SUMMARY_PERIOD = 0
# Max number of checkpoints to keep
_C.TRAIN.MAX_TO_KEEP = 100

# Regex patterns of modules and/or parameters to freeze
_C.TRAIN.FROZEN_PATTERNS = ()

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# Batch size
_C.VAL.BATCH_SIZE = 1
# Period to validate. 0 for disable
_C.VAL.PERIOD = 0
# Period to log validation status. 0 for disable
_C.VAL.LOG_PERIOD = 20
# The metric for best validation performance
_C.VAL.METRIC = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means use time seed.
_C.RNG_SEED = 1
