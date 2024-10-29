
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed_gpu(seed, gpus:list):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    cudnn.benchmark = True
    cudnn.enabled=True
    if torch.cuda.is_available():
      torch.cuda.set_device(gpus[0])
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    return device