#!/usr/bin/env python3

"""Оценка набора данных"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage
from tqdm import tqdm

from arcnn.data import get_valid_dataset
from arcnn.data import DatasetFromFolder, TransformImgToJpg, ReduceSize
from arcnn.utils import AverageMeter
from process import Process
from arcnn import metrics


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        type=str,
        default='ARCNN',
        help='ARCNN or FastARCNN or VDSR'
    )
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = arguments()
    jpeg_quality = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process = Process(opt.arch, opt.model, device)

    # Load dataset
    eval_set = DatasetFromFolder(
        dataset_dir=opt.data_folder,
        source_transform=Compose([
            ReduceSize(5),
            ToTensor(),
        ]),
        input_transform=Compose([
            ToPILImage(),
            TransformImgToJpg(jpeg_quality=jpeg_quality),
            ToTensor()
        ])
    )
    eval_loader = DataLoader(
        dataset=eval_set,
        num_workers=4,
        batch_size=1,
        shuffle=False
    )

    # Eval
    psnr = AverageMeter()
    ssim = AverageMeter()

    eval_bar = tqdm(eval_loader)
    for idx, item in enumerate(eval_bar):
        img_input, img_truth = item  # batch = 1
        img_input = ToPILImage()(img_input[0])

        img_out = process.run(img_input)
        img_out = ToTensor()(img_out).unsqueeze(0)

        psnr.update(metrics.PSNR()(img_out, img_truth).item(), 1)
        ssim.update(metrics.ssim(img_out, img_truth).item(), 1)

    print("Jpeg quality: {}".format(jpeg_quality))
    print("PSNR avg: {:.4f} dB; SSIM avg: {:.4f}".format(psnr.avg, ssim.avg))
