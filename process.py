#!/usr/bin/env python3

import os
import argparse

import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.utils as utils

from arcnn.model import get_model_by_name


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


def image_prepea(filepath) -> torch.Tensor:
    """Открывает и подготавливает файл для отправки в нейронную сеть"""
    img = Image.open(filepath)
    img = img.convert("RGB")  # 3 канала
    # img = ImageOps.autocontrast(img)
    return ToTensor()(img).unsqueeze(0)


if __name__ == "__main__":
    opt = arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # Load model and et
        model = get_model_by_name(opt.arch, opt.model)
        model = model.to(device)
        model.eval()

        pred = model(image_prepea(opt.image).to(device))
        img_out = pred.data.cpu().squeeze(0)
        utils.save_image(img_out, gen_filename(opt.image))
