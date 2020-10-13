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


class Process:

    def __init__(self, arch, model_path, device):
        self.device = device
        self.model = get_model_by_name(arch, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def run(self, pil_img):
        with torch.no_grad():
            w, h = pil_img.size
            img_result = Image.new("RGB", (w, h))
            for img_piece, left, top, w, h in self.image_split(pil_img, 224, 224):
                pred = self.model(self.image_prepea(img_piece).to(self.device))
                img_out = self.image_post(pred.data.cpu().squeeze(0))
                img_result.paste(img_out.crop((0, 0, w, h)), (left, top))
        return img_result

    def image_prepea(self, pil_img) -> torch.Tensor:
        """Открывает и подготавливает файл для отправки в нейронную сеть"""
        img = pil_img.convert("RGB")  # 3 канала
        # img = ImageOps.autocontrast(img)
        return ToTensor()(img).unsqueeze(0)

    def image_post(self, img_tensor):
        """Пост обработка, преобразует тэнзор в изображение"""
        # Magic
        ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        return im

    def image_split(self, pil_img, height=224, weight=224):
        """Режет изображение на кусочки"""
        img_w, img_h = pil_img.size
        for left in range(0, img_w, weight):
            for top in range(0, img_h, height):
                h = (img_h - top) if (top + height) > img_h else height
                w = (img_w - left) if (left + weight) > img_w else weight
                img = Image.new("RGB", (weight, height))
                # crop(l, u, r, l)
                img.paste(
                    pil_img.crop((left, top, left + w, top + h)),
                    (0, 0)
                )
                yield img, left, top, w, h


if __name__ == "__main__":
    opt = arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process = Process(opt.arch, opt.model, device)
    img_in = Image.open(opt.image)
    img_result = process.run(img_in)
    img_result.save(gen_filename(opt.image))

    # with torch.no_grad():
    #     # Load model and et
    #     model = get_model_by_name(opt.arch, opt.model)
    #     model = model.to(device)
    #     model.eval()

    #     img_in = Image.open(opt.image)
    #     w, h = img_in.size
    #     img_result = Image.new("RGB", (w, h))
    #     for img_piece, left, top, w, h in image_split(img_in, 224, 224):
    #         pred = model(image_prepea(img_piece).to(device))
    #         img_out = image_post(pred.data.cpu().squeeze(0))
    #         img_result.paste(img_out.crop((0, 0, w, h)), (left, top))

        # img_result.save(gen_filename(opt.image))
