import os
import os.path as osp
import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas

from lib.flownet import FlowNetC, FlowNetS
from lib.datasets.flow.flyingthings3d import FlyingThings3DTest
from lib.datasets.flow.sintel import SintelTest, SintelFullTrain
from lib.datasets.flow.flyingchairs import FlyingChairsTest
from lib.metrics import compute_flow_metrics
from lib.augmentation import FlowNetAugmentation
from lib.utils import load_model, load_all
from lib.log import Logger

PRINT_INTERVAL = 10
LOG_INTERVAL = 20


def setup_model(args):
    if args.model == 'FlowNetC':
        model = FlowNetC(C=args.C, cuda_corr=args.cuda_corr)
    elif args.model == 'FlowNetS':
        model = FlowNetS(C=args.C)

    model.cuda()
    model.train()

    return model


def setup_preprocessing():
    preprocess_fct = FlowNetAugmentation()
    preprocess_fct.out_size = None  # upscale to next height and width that are divisible by 64
    preprocess_fct.spatial_aug_prob = 0.  # only resize, no cropping
    preprocess_fct.color_aug_prob = 0.  # no color augmentation
    preprocess_fct.augment_image_only = True  # do not resize the ground truth
    return preprocess_fct


def setup_dataloader(args):
    global LOG_INTERVAL

    if args.dataset == 'FlyingThings3D':
        dataset_cls = FlyingThings3DTest
        LOG_INTERVAL = 100
    elif args.dataset == 'FlyingChairs':
        dataset_cls = FlyingChairsTest
        LOG_INTERVAL = 10
    elif args.dataset == 'Sintel':
        dataset_cls = SintelTest
        LOG_INTERVAL = 5
    elif args.dataset == 'SintelFull':
        dataset_cls = SintelFullTrain
        LOG_INTERVAL = 20

    preprocess_fct = setup_preprocessing()
    dataloader = dataset_cls.init_as_loader(batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                            drop_last=False, aug_fcts=preprocess_fct)
    return dataloader


def restore(model, checkpoint_path):
    print("Restoring model weights from {}".format(checkpoint_path))
    if "checkpoint-model" in checkpoint_path:
        load_model(path=checkpoint_path, model=model, strict=True)
    else:
        load_all(path=checkpoint_path, model=model)
    print()


def setup_logger(log_dir, model):
    writer = SummaryWriter(log_dir=log_dir, comment='test')
    logger = Logger(writer, model=model)
    return logger


def run_model(model, sample):
    image_list = sample['images']
    image_1 = image_list[0]
    image_2 = image_list[1]

    model_output_dict = model(image_1, image_2)

    return model_output_dict


def log_metrics(metrics, values, sample_idx):
    for k, v in values.items():
        metrics.loc[sample_idx, k] = v


def sample_to_cuda(data, device=None):
    if isinstance(data, dict):
        return {key: sample_to_cuda(data[key], device) for key in data.keys()}
    elif isinstance(data, list):
        return [sample_to_cuda(val, device) for val in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def print_info(args):
    print("Initializing evaluation.")
    print("\tModel: {} (feature dimensionality: {})".format(args.model, args.C))
    print("\tEvaluation dataset: {}".format(args.dataset))

    if args.restore is not None:
        print("\tRestore from checkpoint: {}".format(args.restore))
    else:
        print("\tWarning: No checkpoint given. Evaluation is done with an untrained model!")

    print("\tOutput directory: {}".format(args.output))
    print()


def eval(args):

    print_info(args=args)

    out_base = args.output
    eval_base = osp.join(out_base, 'eval')
    eval_dir = osp.join(eval_base, args.dataset)
    log_dir = osp.join(eval_dir, 'logs')
    os.makedirs(eval_base, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = setup_model(args=args)
    dataloader = setup_dataloader(args=args)

    if args.restore is not None:
        restore(model=model, checkpoint_path=args.restore)

    logger = setup_logger(log_dir=log_dir, model=model)

    metrics = pandas.DataFrame()
    metrics.index.name = "sample"
    metrics.columns.name = "metric"

    start = time.time()
    for sample_idx, sample in enumerate(dataloader):

        sample = sample_to_cuda(sample)

        model_output = run_model(model=model, sample=sample)
        metric_values, qualitatives = compute_flow_metrics(sample=sample, model_output=model_output)
        log_metrics(metrics=metrics, values=metric_values, sample_idx=sample_idx)

        if (sample_idx > 0) and (sample_idx % PRINT_INTERVAL == 0):
            end = time.time()
            time_per_iteration = (end - start) / PRINT_INTERVAL
            print("Sample {}/{} - {:.2f}s per sample - metrics:".format(sample_idx, len(dataloader), time_per_iteration))
            print("\t" + metrics.loc[sample_idx].to_string().replace("\n", "\n\t"))
            start = time.time()

        if sample_idx % LOG_INTERVAL == 0:
            logger.log_eval(sample=sample, model_output=model_output, metrics=metric_values,
                            qualitatives=qualitatives, sample_idx=sample_idx)

    results = metrics.mean()
    metrics.to_csv(osp.join(eval_dir, "metrics.csv"))
    results.to_csv(osp.join(eval_dir, "results.csv"))
    print(results.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for evaluation outputs.")
    parser.add_argument('--model', default="FlowNetC", help="FlowNetC or FlowNetS model.")
    parser.add_argument('--restore', help="Path to a checkpoint to restore from.")
    parser.add_argument('--dataset', default="FlyingThings3D", help="Dataset for evaluation. Options: FlyingChairs, FlyingThings3D, Sintel, SintelFull.")
    parser.add_argument('--C', default=48, type=int, help="Base feature dimensionality. Original paper uses C=64. In the exercise we use C=48.")
    parser.add_argument('--cuda_corr', action='store_true', help='Use CUDA correlation layer in FlowNetC.')
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)

    eval(args)
