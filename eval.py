#!/usr/bin/env python3

"""Оценка набора данных"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage

from arcnn.model import get_model_by_name
from arcnn.data import get_valid_dataset
from arcnn.data import DatasetFromFolder, TransformImgToJpg, ReduceSize
from train import validation


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

    model = get_model_by_name(opt.arch, opt.model)
    model = model.to(device)

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
    criterion = nn.MSELoss()
    _, _, psnr_avg, ssim_avg = validation(
        model, eval_loader, criterion, device, False)
    print("Jpeg quality: {}".format(jpeg_quality))
    print("PSNR avg: {:.4f} dB; SSIM avg: {:.4f}".format(psnr_avg, ssim_avg))
