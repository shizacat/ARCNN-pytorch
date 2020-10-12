import os
import io

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
)


class TransformImgToJpg:
    def __init__(self, jpeg_quality=1):
        self.jpeg_quality = jpeg_quality
    
    def __call__(self, pil_img):
        buffer = io.BytesIO()
        pil_img.save(buffer, format='jpeg', quality=self.jpeg_quality)
        buffer.seek(0)
        img = Image.open(buffer)
        return img


class DatasetFromFolder(Dataset):
    """
        source - Изображение, которое будет считать оригиналом
        input - Изображение, которое будет подаваться на вход нейронной сети

        На вход принимает только png файлы
    """
    def __init__(self, dataset_dir, source_transform, input_transform):
        super().__init__()
        self.image_filenames = []
        self.source_transform = source_transform
        self.input_transform = input_transform

        for x in os.listdir(dataset_dir):
            if not self.is_image_file(x):
                continue
            self.image_filenames.append(os.path.join(dataset_dir, x))

    def __getitem__(self, index):
        src_image = self.source_transform(Image.open(self.image_filenames[index]))
        inp_image = self.input_transform(src_image)
        return inp_image, src_image

    def __len__(self):
        return len(self.image_filenames)
    
    def is_image_file(self, filename):
        fmt = ['.png']
        return any(filename.lower().endswith(extension) for extension in fmt)


class ReduceSize:
    """Уменьшить размер в k раз"""
    def __init__(self, k):
        self.k = k
    
    def __call__(self, pil_img):
        img_width, img_height = pil_img.size
        return pil_img.resize(
            (int(img_width/self.k), int(img_height/self.k)), Image.BICUBIC)


def get_train_dataset(dataset_dir, crop_size):
    ds = DatasetFromFolder(
        dataset_dir=dataset_dir,
        source_transform=Compose([
            ReduceSize(5),
            RandomCrop(crop_size),
            ToTensor(),
        ]),
        input_transform=Compose([
            ToPILImage(),
            TransformImgToJpg(jpeg_quality=30),
            ToTensor()
        ])
    )
    return ds


def get_valid_dataset(dataset_dir, crop_size):
    ds = DatasetFromFolder(
        dataset_dir=dataset_dir,
        source_transform=Compose([
            ReduceSize(5),
            CenterCrop(crop_size),
            ToTensor(),  # C, H, W
        ]),
        input_transform=Compose([
            ToPILImage(),
            TransformImgToJpg(jpeg_quality=30),
            ToTensor()
        ])
    )
    return ds
