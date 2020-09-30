#!/usr/bin/env python3

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from arcnn.model import get_model_by_name
from arcnn.data import get_train_dataset, get_valid_dataset
from arcnn.utils import AverageMeter
from arcnn import metrics


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument(
        '--arch',
        type=str,
        default='ARCNN',
        help='ARCNN or FastARCNN or VDSR'
    )
    parser.add_argument(
        "--save_result",
        default=10,
        type=int,
        help='Save result after n epochs'
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument(
        '--lr', type=float, default=5e-4, help="Learning rate, default 5e-4"
    )
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    opt = parser.parse_args()
    return opt


def check_folder(opt):
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)
    
    results = os.path.join(opt.outputs_dir, "results")
    epochs = os.path.join(opt.outputs_dir, "epochs")
    if not os.path.exists(results):
        os.makedirs(results)
    if not os.path.exists(epochs):
        os.makedirs(epochs)


def train_one_epoch(model, optimizer, train_loader, criterion):
    """
        Return: 
            Loss_avg
    """
    model.train()
    epoch_losses = AverageMeter()
    train_bar = tqdm(train_loader)
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        preds = model(inputs)
        loss = criterion(preds, labels)
        epoch_losses.update(loss.item(), len(inputs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_description(
            desc="Train [{:04}/{:04}] Loss avg: {:.4f}".format(
                epoch, opt.num_epochs, epoch_losses.avg
            )
        )
    return epoch_losses.avg


def validation(model, val_loader, criterion, device, get_images=False):
    images = []

    model.eval()
    psnr = AverageMeter()
    ssim = AverageMeter()
    loss = AverageMeter()
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for idx, item in enumerate(val_bar):
            inputs, labels = item
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
    
            loss.update(criterion(preds, labels).item(), len(inputs))
            psnr.update(metrics.PSNR()(preds, labels).item(), len(inputs))
            ssim.update(metrics.ssim(preds, labels).item(), len(inputs))

            val_bar.set_description(
                desc='Valid [{:04}/{:04}] Loss: {:.4f}; PSNR: {:.4f} dB; SSIM: {:.4f}'.format(
                    idx, len(val_loader), loss.avg, psnr.avg, ssim.avg
                )
            )
            
            if get_images:
                images.extend([
                    labels.data.cpu().squeeze(0),
                    inputs.data.cpu().squeeze(0),
                    preds.data.cpu().squeeze(0)
                ])

    return images, loss.avg, psnr.avg, ssim.avg


def save_images(images: list, path: str, epoch: int):
    """
    Args:
        images: List[Tuple[img, img, img]]
    """
    images = torch.stack(images)
    images = torch.chunk(images, images.size(0) // 15)
    val_save_bar = tqdm(images, desc='Saving validation results')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(
            image,
            os.path.join(
                path,
                "epoch_{:04}_index_{:02}.png".format(epoch, index)
            ),
            padding=5
        )
        index += 1


if __name__ == "__main__":
    opt = arguments()
    check_folder(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt.seed)

    # Load model and et
    print("Load model:", opt.arch)
    model = get_model_by_name(opt.arch, opt.model)
    model = model.to(device)

    criterion = nn.MSELoss()

    if opt.arch in ["ARCNN", "FastARCNN"]:
        optimizer = optim.Adam([
            {'params': model.base.parameters()},
            {'params': model.last.parameters(), 'lr': opt.lr * 0.1},
        ], lr=opt.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Load dataset
    train_set = get_train_dataset(
        os.path.join(opt.data_folder, "train"),
        crop_size=600
    )
    val_set = get_valid_dataset(
        os.path.join(opt.data_folder, "val"),
        (600, 600)
    )
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=1,
        shuffle=False
    )

    # Train
    results = {'train_loss': [], 'valid_loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, opt.num_epochs + 1):
        train_loss_avg = train_one_epoch(
            model, optimizer, train_loader, criterion
        )
        images, val_loss, psnr, ssim = validation(
            model, val_loader, criterion, device, (epoch % opt.save_result) == 0
        )

        # Save images result
        if images:
            save_images(images, os.path.join(opt.outputs_dir, "results"), epoch)
        
        # Save statistics
        results["train_loss"].append(train_loss_avg)
        results["valid_loss"].append(val_loss)
        results["psnr"].append(psnr)
        results["ssim"].append(ssim)
        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={
                    'Loss Train Avg': results['train_loss'],
                    'Loss Valid Avg': results['valid_loss'],
                    'PSNR Avg': results['psnr'],
                    'SSIM Avg': results['ssim']
                },
                index=range(1, epoch + 1)
            )
            data_frame.to_csv(
                os.path.join(opt.outputs_dir, "train_staticstics.csv"),
                index_label='Epoch'
            )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(
                opt.outputs_dir,
                "epochs",
                "{}_epoch_{:04}.pth".format(opt.arch, epoch)
            )
        )
