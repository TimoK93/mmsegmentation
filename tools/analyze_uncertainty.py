# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def main(args):

    samples = [
        x for x in os.listdir(os.path.join(args.path))
        if x.endswith('.npy') and not x.startswith('.')
    ]

    num_pixels, num_errors = 0, 0
    hists, error_hists = list(), list()
    for s in samples:
        # Load data
        uncertainty = np.load(os.path.join(args.path, s))
        prediction = np.array(
            Image.open(
                os.path.join(
                    args.path,
                    s.replace('_raw_uncertainty.npy', '_prediction.png'))))
        gt = np.array(
            Image.open(
                os.path.join(args.path,
                             s.replace('_raw_uncertainty.npy', '_gt.png'))))
        # Calculate metrics
        uncertainty = np.power(uncertainty, 0.1)  # Amplify small values
        uncertainty = 1 - uncertainty  # Convert uncertainty to certainty
        ignore = gt == args.ignore_label
        errors = (prediction != gt) * (ignore == 0)
        num_pixels += np.sum(1 - ignore)
        num_errors += np.sum(errors)
        bin_counts, bin_edges = np.histogram(uncertainty[~ignore], 100000,
                                             (0, 1))
        acc_bin_counts = np.cumsum(bin_counts)
        hists.append((acc_bin_counts, bin_edges))
        bin_counts, bin_edges = np.histogram(uncertainty[errors], 100000,
                                             (0, 1))
        acc_bin_counts = np.cumsum(bin_counts)
        error_hists.append((acc_bin_counts, bin_edges))

    total_hist, total_error_hist = \
        np.zeros_like(hists[0][0]), np.zeros_like(error_hists[0][0])
    for (h, _), (e, _) in zip(hists, error_hists):
        total_hist += h
        total_error_hist += e
    total_area = total_hist / num_pixels
    total_accuracy = 1 - (num_errors - total_error_hist) / num_pixels

    # Plot file
    file_name = os.path.join(args.path, 'error_detection.pdf')
    plt.rcParams['figure.figsize'] = [7.00, 3.50]
    plt.rcParams['figure.autolayout'] = True
    plt.gca().set_ylabel('Pixel Level Accuracy')
    plt.gca().set_xlabel('Relative Area marked by most uncertain pixel')
    plt.gca().set_title('Error Detection Accuracy')
    plt.plot(total_area, total_accuracy)
    plt.savefig(file_name)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates the model uncertainty')
    parser.add_argument('--path', type=str, help='The directory to load files')
    parser.add_argument(
        '--ignore-label', default=255, type=int, help='Ignore label')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
