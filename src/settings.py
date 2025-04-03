#!/usr/bin/env python3

IMAGE_SIZE = (256, 256)
DATA_BASE_DIR = "../data"
DELTA = 256

NUM_EPOCHS = 500

# transformer hyperparams

EMBED_DIM = 512
PATCH_SIZE = 16
DEPTH = 6
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.1
IN_CHANS = 3
