import os
import os.path as osp
import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from lib.flownet import FlowNetC, FlowNetS
from lib.datasets.flow.flyingthings3d import FlyingThings3DTrain
from lib.datasets.flow.sintel import SintelTrain
from lib.loss import FlowLoss, PhotometricLoss
from lib.augmentation import FlowNetAugmentation
from lib.utils import TrainStateSaver, WeightsOnlySaver
from lib.log import Logger

PRINT_INTERVAL = 100
LOG_INTERVAL = 5000
LOG_LOSS_INTERVAL = 200
CHECKPOINT_INTERVAL = 100000


def setup_model(args):

    if args.model == 'FlowNetC':
        model = FlowNetC(C=args.C, cuda_corr=args.cuda_corr)
    elif args.model == 'FlowNetS':
        model = FlowNetS(C=args.C)

    model.cuda()
    model.train()

    return model


def setup_augmentation():
    aug_fct = FlowNetAugmentation()
    return aug_fct


def setup_dataloader(args):

    if args.dataset == 'FlyingThings3D':
        dataset_cls = FlyingThings3DTrain
    elif args.dataset == 'Sintel':
        dataset_cls = SintelTrain

    aug_fct = setup_augmentation()
    dataloader = dataset_cls.init_as_loader(batch_size=4, shuffle=True, num_workers=8, pin_memory=True,
                                            drop_last=True, aug_fcts=aug_fct)
    return dataloader


def setup_loss(model, args):
    if not args.photometric:
        loss_fct = FlowLoss(model)
    else:
        loss_fct = PhotometricLoss(model, use_smoothness_loss=args.smoothness_loss)
    return loss_fct


def setup_optimization(model, finished_iterations):

    lr_base = 1e-4
    lr_intervals = [300000, 400000, 500000]
    gamma = 0.5

    params = [p for p in model.parameters() if p.requires_grad]

    print("Total number of parameters to optimize: {}\n".format(sum(p.numel() for p in params)))

    optimizer = torch.optim.Adam(params, lr=lr_base)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_intervals, gamma=gamma)
    for _ in range(finished_iterations):
        scheduler.step()

    return optimizer, scheduler


def setup_savers(model, optimizer, scheduler, checkpoint_dir):
    saver_all = TrainStateSaver(model=model, optim=optimizer, scheduler=scheduler, base_path=checkpoint_dir,
                                base_name="checkpoint-train", max_to_keep=3)

    saver_weights_only = WeightsOnlySaver(model=model, base_path=checkpoint_dir, base_name="checkpoint-model")

    return saver_all, saver_weights_only


def restore(saver_all, saver_weights_only, checkpoint_path):
    if "checkpoint-model" in checkpoint_path:
        saver_weights_only.load(full_path=checkpoint_path)
    else:
        saver_all.load(full_path=checkpoint_path)
    print()


def setup_logger(log_dir, model, optimizer):
    writer = SummaryWriter(log_dir=log_dir, comment='train')
    logger = Logger(writer, model=model, optimizer=optimizer)
    return logger


def run_model(model, sample):

    image_list = sample['images']
    image_1 = image_list[0]
    image_2 = image_list[1]

    model_output_dict = model(image_1, image_2)

    return model_output_dict


def sample_to_cuda(data, device=None):
    if isinstance(data, dict):
        return {key: sample_to_cuda(data[key], device) for key in data.keys()}
    elif isinstance(data, list):
        return [sample_to_cuda(val, device) for val in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def train(args):

    print("Initializing training.")
    print("\tModel: {} (feature dimensionality: {})".format(args.model, args.C))
    print("\tIterations: {}".format(args.iterations))
    print("\tTraining dataset: {}".format(args.dataset))
    print("\tSupervised training: {}".format(not args.photometric))
    if args.photometric:
        print("\tUsing smoothness loss: {}".format(args.smoothness_loss))
    print("\tOutput directory: {}".format(args.output))
    if args.restore is not None:
        print("\tRestore from checkpoint: {}".format(args.restore))
    if args.completed_iterations is not None:
        print("\tAlready completed iterations: {}".format(args.completed_iterations))
    print()

    out_base = args.output
    checkpoint_dir = osp.join(out_base, 'checkpoints')
    log_dir = osp.join(out_base, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    max_iteration = args.iterations
    finished_iterations = 0 if args.completed_iterations is None else args.completed_iterations
    epochs = -1

    model = setup_model(args=args)
    dataloader = setup_dataloader(args=args)
    loss_fct = setup_loss(model=model, args=args)
    optimizer, scheduler = setup_optimization(model=model, finished_iterations=finished_iterations)
    saver_all, saver_weights_only = setup_savers(model=model, optimizer=optimizer, scheduler=scheduler,
                                                 checkpoint_dir=checkpoint_dir)
    
    if args.restore is not None:
        restore(saver_all=saver_all, saver_weights_only=saver_weights_only, checkpoint_path=args.restore)

    logger = setup_logger(log_dir=log_dir, model=model, optimizer=optimizer)

    start = time.time()
    while finished_iterations < max_iteration:
        epochs += 1
        for iter_in_epoch, sample in enumerate(dataloader):

            sample = sample_to_cuda(sample)

            optimizer.zero_grad()
            model_output = run_model(model=model, sample=sample)
            loss, sub_losses, pointwise_losses = loss_fct(sample=sample, model_output=model_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()

            finished_iterations += 1

            if finished_iterations % PRINT_INTERVAL == 0:
                end = time.time()
                time_per_iteration = (end - start) / PRINT_INTERVAL
                print("Iteration {}/{} - {:.2f}s per iteration - loss: {:.5f}".format(finished_iterations,
                                                                                       max_iteration,
                                                                                       time_per_iteration, loss))
                start = time.time()

            if finished_iterations % LOG_INTERVAL == 0:
                logger.log(sample=sample, model_output=model_output, loss=loss, sub_losses=sub_losses,
                           pointwise_losses=pointwise_losses, step=finished_iterations)
            elif finished_iterations % LOG_LOSS_INTERVAL == 0:
                logger.log(sample=sample, model_output=model_output, loss=loss, sub_losses=sub_losses,
                           pointwise_losses=pointwise_losses, step=finished_iterations, loss_only=True)

            if finished_iterations % CHECKPOINT_INTERVAL == 0:
                save_path = saver_all.save(iter=finished_iterations)
                print("Saving train state to {}.".format(save_path))

            if finished_iterations >= max_iteration:
                save_path = saver_weights_only.save(iter=finished_iterations)
                print("Saving model weights to {}.".format(save_path))
                break

    epochs += iter_in_epoch / len(dataloader)
    print("Trained for {} iterations (={:.2f} epochs).".format(finished_iterations, epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data like logs and model weights.")
    parser.add_argument('--iterations', default=600000, type=int, help="Number of training iterations.")
    parser.add_argument('--model', default="FlowNetC", help="FlowNetC or FlowNetS model.")
    parser.add_argument('--completed_iterations', type=int, help="Already completed number of iterations.")
    parser.add_argument('--restore', help="Path to a checkpoint to restore from.")
    parser.add_argument('--dataset', default="FlyingThings3D", help="Dataset for training. Options: FlyingThings3D, Sintel.")
    parser.add_argument('--C', default=48, type=int, help="Base feature dimensionality. Original paper uses C=64. In the exercise we use C=48.")
    parser.add_argument('--cuda_corr', action='store_true', help='Use CUDA correlation layer in FlowNetC.')
    parser.add_argument('--photometric', action='store_true', help='Train self-supervised using the photometric loss.')
    parser.add_argument('--smoothness_loss', action='store_true', help='Use the smoothness loss in addition to the photometric loss.')
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)

    train(args)
