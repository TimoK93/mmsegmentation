# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

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
    for batch_indices, data in zip(loader_indices, data_loader):
        x, _ = scatter_kwargs(
            inputs=data, kwargs=None, target_gpus=model.device_ids)
        if args.aug_test:
            raise NotImplementedError
        else:
            logits, unc = \
                model.module.simple_test_logits_with_uncertainty(**x[0])
        pred = logits.argmax(axis=1)
        uncertainty = unc.squeeze()
        pred = pred.squeeze()
        heatmap = (255 * (uncertainty / uncertainty.max())).astype(np.uint8)
        img_info = dataset.img_infos[batch_indices[0]]
        file_name = os.path.join(
            tmpdir, img_info['ann']['seg_map'].split(os.path.sep)[-1])

        Image.fromarray(heatmap).save(file_name)
        if args.store_raw_uncertainty:
            new_file_name = file_name + '_raw_uncertainty.numpy'
            np.save(new_file_name, uncertainty.astype(np.float16))
        if args.store_prediction:
            new_file_name = file_name + '_prediction.png'
            Image.fromarray(pred.astype(np.uint8)).save(new_file_name)
        if args.store_ground_truth:
            gt_file_name = os.path.join(
                dataset.ann_dir,
                dataset.img_infos[batch_indices[0]]['ann']['seg_map'])
            new_file_name = file_name + '_gt.png'
            shutil.copyfile(gt_file_name, new_file_name)
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model prediction with uncertainty result')
    parser.add_argument(
        '--config', type=str, help='ensemble config files path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control aug-result or single-result (default)')
    parser.add_argument(
        '--store-raw-uncertainty',
        action='store_true',
        help='stores additional uncertainty in float16 format')
    parser.add_argument(
        '--store-prediction',
        action='store_true',
        help='stores additional rgb images')
    parser.add_argument(
        '--store-ground-truth',
        action='store_true',
        help='stores additional ground truth files')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu to use')

    args = parser.parse_args()
    assert args.out, "uncertainty result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
