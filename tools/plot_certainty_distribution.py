# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


@torch.no_grad()
def main(args):
    cfg = mmcv.Config.fromfile(args.config)

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None):
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    torch.cuda.empty_cache()
    tmpdir = args.out
    mmcv.mkdir_or_exist(tmpdir)
    model = MMDataParallel(model, device_ids=[args.gpu])
    model.eval()

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    hists = list()
    for batch_indices, data in zip(loader_indices, data_loader):
        x, _ = scatter_kwargs(
            inputs=data, kwargs=None, target_gpus=model.device_ids)
        if args.aug_test:
            pred = model.module.aug_test_logits(**x[0])
        else:
            pred = model.module.simple_test_logits(**x[0])
        pred = pred.max(axis=1).squeeze()
        hists.append(np.histogram(pred, 100, (0, 1)))

        prog_bar.update()

    values, bins = None, hists[0][1]
    for v, b in hists:
        values = v if values is None else values + v
    values = values / np.sum(values)
    mean_certainty = np.sum(values * bins[1:])

    # Plot file
    file_name = os.path.join(tmpdir, args.name + '.pdf')
    plt.rcParams['figure.figsize'] = [7.00, 3.50]
    plt.rcParams['figure.autolayout'] = True
    plt.gca().set_ylabel('Rel. Occurrence')
    plt.gca().set_xlabel('Certainty')
    plt.gca().set_title(f'Histogramm of Certainty (mean certainty: '
                        f'{mean_certainty:.4f})')
    plt.plot(bins[1:], values)
    plt.savefig(file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plots the mean certainty over the dataset')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control ensemble aug-result or single-result (default)')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu to use')
    parser.add_argument(
        '--name',
        type=str,
        default='certainty_dist',
        help='name for the result file')

    args = parser.parse_args()
    assert args.out, "result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
