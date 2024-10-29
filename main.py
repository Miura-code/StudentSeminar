import os
import numpy as np
from config.main_config import MainConfig
from utils.logging import get_std_logging

from torch.utils.tensorboard import SummaryWriter

def main():
    config.print_params(logger.info)
    logger.info("-> ログを開始")
    logger.debug("-> デバックのためのログ")

    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)
    
    data = np.linspace(0, 100, 300)
    for epoch, d in enumerate(data):
        writer.add_scalar('train/data', d, epoch)
    writer.close()


if __name__ == "__main__":
    config = MainConfig()
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.save)))
    main()