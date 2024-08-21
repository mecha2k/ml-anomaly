import torch
import argparse
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from tqdm import tqdm
from pathlib import Path
from parse_config import ConfigParser
from utils import prepare_device
from submissions import final_submission


def get_data_loader(training=True):
    return getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=config["data_loader"]["args"]["batch_size"],
        shuffle=True if training else False,
        validation_split=0.0,
        training=training,
        window_size=config["data_loader"]["args"]["window_size"],
        window_given=config["data_loader"]["args"]["window_given"],
        stride=config["data_loader"]["args"]["stride"],
        init_loader=False,
    )


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)

    # build model architecture
    model = config.init_obj("arch", module_arch, n_tags=train_loader.train_df.shape[1])
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    model.eval()

    train_errors = []
    with torch.no_grad():
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            errors = torch.mean(torch.abs(targets - outputs), dim=1).cpu().numpy()
            train_errors.extend(errors)
    threshold = np.mean(train_errors) + 2 * np.std(train_errors)
    print(f"Threshold based on training data : {threshold}")

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    with torch.no_grad():
        for dataset in test_loader:
            inputs = dataset["input"].to(device)
            targets = dataset["target"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(outputs, targets) * batch_size

    n_samples = len(test_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    )
    logger.info(log)

    data_path = Path(config["data_loader"]["args"]["data_dir"])
    final_submission(model, test_loader, threshold, device, data_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default=None, type=str, help="config file path (default: None)"
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)"
    )

    config = ConfigParser.from_args(args)
    main(config)
