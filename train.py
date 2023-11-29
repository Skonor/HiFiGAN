import argparse
import collections
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config["arch"]["generator"], module_arch)
    logger.info(generator)

    discriminator = config.init_obj(config["arch"]["discriminator"], module_arch)
    logger.info(discriminator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module_G = config.init_obj(config["loss"]["loss_G"], module_loss).to(device)
    loss_module_D = config.init_obj(config["loss"]["loss_D"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_G = filter(lambda p: p.requires_grad, generator.parameters())
    trainable_params_D = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_G = config.init_obj(config["optimizers"]["optimizer_G"], torch.optim, trainable_params_G)
    optimizer_D = config.init_obj(config["optimizers"]["optimizer_D"], torch.optim, trainable_params_D)
    lr_scheduler_G = config.init_obj(config["lr_scheduler"]["lr_scheduler_G"], torch.optim.lr_scheduler, optimizer_G)
    lr_scheduler_D = config.init_obj(config["lr_scheduler"]["lr_scheduler_D"], torch.optim.lr_scheduler, optimizer_D)

    trainer = Trainer(
        generator,
        discriminator,
        loss_module_G,
        loss_module_D,
        optimizer_G,
        optimizer_D,
        metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_G=lr_scheduler_G,
        lr_scheduler_D=lr_scheduler_D,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-f",
        "--finetune",
        default=None,
        type=str,
        help="path to pretrained model checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
