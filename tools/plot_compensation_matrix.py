# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Compensation Matrix',
                          color_theme='winter'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """

    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(num_classes, num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Class i', fontdict=label_font)
    plt.xlabel('Class j', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                round(confusion_matrix[i, j], 1),
                ha='center',
                va='center',
                color='w',
                size=10)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'compensation_matrix.pdf'), format='pdf')
    if show:
        plt.show()


@torch.no_grad()
def main(args):

    cfg = mmcv.Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None):
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    torch.cuda.empty_cache()
    tmpdir = args.out
    mmcv.mkdir_or_exist(tmpdir)
    model = MMDataParallel(model, device_ids=[args.gpu])
    model.eval()

    compensation = model.module.decode_head.compensations.\
        weight.data.clone().cpu().numpy()[:, :, 0, 0]
    plot_confusion_matrix(compensation, dataset.CLASSES, save_dir=tmpdir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model prediction with uncertainty result')
    parser.add_argument(
        '--config', type=str, help='ensemble config files path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu to use')

    args = parser.parse_args()
    assert args.out, "uncertainty result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
