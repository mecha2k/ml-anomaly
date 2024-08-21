import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=256,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1,
        window_size=41,
        window_given=40,
        stride=10,
        init_loader=False,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch, n_tags=data_loader.train_df.shape[1])
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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for dataset in tqdm(data_loader):
            inputs = dataset["input"].to(device)
            targets = dataset["target"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(outputs, targets) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    )
    logger.info(log)


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
