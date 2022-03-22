import torch


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NORMALIZATION = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
