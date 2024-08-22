import argparse

from train import main as train_main
from test import main as test_main
from parse_config import ConfigParser, _update_config


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default="config.json", type=str, help="config file path (default: None)"
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)"
    )

    config = ConfigParser.from_args(args)
    config.resume = train_main(config)
    test_main(config)
