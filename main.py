import os
from config.main_config import MainConfig
from utils.logging import get_std_logging


def main():
    config.print_params(logger.info)
    logger.info("-> ログを開始")
    logger.debug("-> デバックのためのログ")

if __name__ == "__main__":
    config = MainConfig()
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.save)))
    main()