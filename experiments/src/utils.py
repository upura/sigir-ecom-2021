import os
import random
import numpy as np


def seed_everything(seed: int = 71, gpu_mode: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if gpu_mode:
        import tensorflow as tf
        import torch

        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
