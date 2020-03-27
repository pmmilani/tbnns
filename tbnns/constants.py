#---------------------------------- constants.py ---------------------------------------#
"""
This file contains all global constants, for easy access
"""

# number of input invariants used for the model
NUM_FEATURES = 15

# number of form invariant basis needed for the diffusivity
NUM_BASIS = 6

# batch size to use for doing a forward pass. Should be a large
# number for speed; does not change the overall result
TEST_BATCH_SIZE = 10000

# Turbulent Prandtl number used by default
PR_T = 0.85

# Number of standard deviations, by default, to clean the features.
# The features at test time are normalized by the mean and standard
# deviation seen at training time; if any point has features more
# than N_STD or less than -N_STD, it means it is very different from
# points seen at training time and therefore won't be used
N_STD = 25

# Minimum value of gamma (or 1/Pr_t) when we clean the tensorial diffusivity
# produced and when we infer gamma from the LES u'c' data. Higher values are
# more stable, but more intrusive. Must be greater than zero.
GAMMA_MIN = 0.02

# Default values for some entries of the FLAGS dictionary, passed 
# in to determine the model
NUM_LAYERS = 8
NUM_NEURONS = 30
NUM_EPOCHS = 0 # how many epochs to run by default (here, run zero)
EARLY_STOP_DEV = 0 # early_stop_dev; 0 deactivates it
TRAIN_BATCH_SIZE = 50 # batch size for training
EVAL_EVERY = 1000 # every this many iterations, check losses
LEARNING_RATE = 1e-3 # default learning rate for Adam
LOSS_TYPE = 'log' # loss_type
REG_FACTOR = 1e-2 # regularization strength
C_PSD = 0.1 # factor for the PSD loss (J_PSD)
C_PRT = 0 # factor for the Pr_t loss (J_PRT)
C_NEG = 0 # factor for the negative diffusivity loss (J_NEG)
DROP_PROB = 0.2 # dropout probability; 0 deactivates it
ENFORCE_PRT = False # whether a separate prediction for Turbulent Prandtl number 
                    # (or gamma) is passed in and the predicted D must obey it.

