#!/usr/bin/env python3

import os
import argparse

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.utils as utils

from lib.model import ARCNN, FastARCNN


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        type=str,
        default='ARCNN',
        help='ARCNN or FastARCNN'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="Path to image"
    )
    opt = parser.parse_args()
    return opt


def gen_filename(filepath) -> str:
    """Генерирует новое имя для файла"""
    path, name = os.path.split(filepath)
    name, ext = os.path.splitext(name)
    return os.path.join(path, "out_{}.png".format(name))


if __name__ == "__main__":
    opt = arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and et
    if opt.arch == 'ARCNN':
        model = ARCNN()
    elif opt.arch == 'FastARCNN':
        model = FastARCNN()

    model = model.to(device)
    model.load_state_dict(torch.load(opt.model))

    img_input = Image.open(opt.image).convert("RGB")
    pred = model(ToTensor()(img_input).unsqueeze(0).to(device))
    img_out = pred.data.cpu().squeeze(0)
    utils.save_image(img_out, gen_filename(opt.image))
