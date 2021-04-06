# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from functools import partial
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.CrossEntropyLoss,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # loss = criterion(samples, outputs, targets)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if args.distributed:
        model = model.module

    # switch to evaluation mode
    model.eval()

    for i, meta_batch in enumerate(
            metric_logger.log_every(data_loader, 100, header)):

        im_s, y_s = meta_batch['train'][0][0].to(
            device, non_blocking=True), meta_batch['train'][1][0].to(device)
        im_q, y_q = meta_batch['test'][0][0].to(
            device, non_blocking=True), meta_batch['test'][1][0].to(device)

        encoder = model.forward_features
        acc1 = LR(encoder, im_s, y_s, im_q, y_q)

        metric_logger.meters['acc1'].update(acc1.item(), n=1)

        if i >= args.num_episodes:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def LR(encoder, support_x, support_ys, query_x, query_ys, norm=False):
    """logistic regression classifier"""
    import sklearn.linear_model
    from torch.nn.functional import normalize
    import torchmetrics

    support = encoder(support_x)
    query = encoder(query_x)
    if norm:
        support = normalize(support)
        query = normalize(query)

    clf = sklearn.linear_model.LogisticRegression(random_state=0,
                                                  solver='lbfgs',
                                                  max_iter=1000,
                                                  C=1,
                                                  multi_class='multinomial')
    support_features_np = support.data.cpu().numpy()
    support_ys_np = support_ys.data.cpu().numpy()
    clf.fit(support_features_np, support_ys_np)

    query_features_np = query.data.cpu().numpy()
    query_ys_pred = clf.predict(query_features_np)

    pred = torch.from_numpy(query_ys_pred).to(support.device,
                                              non_blocking=True)
    return torchmetrics.functional.accuracy(pred, query_ys.long())
