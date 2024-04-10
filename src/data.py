#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype, Normalize, Resize
from torchvision.transforms import functional as F_vision
import datasets


def imagenet(path: Union[str, Path], **kwargs):
    return DataLoader(ImageNet(path), **kwargs)


class ImageNet(Dataset):
    def __init__(self, directory, image_size=224):
        self.set = datasets.Dataset.load_from_disk(str(directory))
        self.pre_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop([image_size, image_size]),
        ])
        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, i):
        image_label = self.set[i]
        image, label = image_label["image"], image_label["label"]
        # Data preprocess
        image = self.pre_transform(image)
        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        image = F_vision.to_tensor(image)
        # Data postprocess
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = self.post_transform(image)
        return image, label

    def __len__(self):
        return len(self.set)


def tiny_imagenet(path: Union[str, Path], **kwargs):
    trainset = TinyImageTrainNet(Path(path) / "train")
    testset = TinyImageTestNet(Path(path) / "val")
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, **kwargs)
    return trainloader, testloader


def preprocess_image(
    image_path: Union[str, Path],
) -> np.ndarray:
    image = cv2.imread(str(image_path))

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([224, 224])(image)
    # Convert image data to pytorch format data
    tensor = F_vision.to_tensor(image)
    # Convert the to the given dtype and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)

    return tensor.numpy()


class TinyImageTrainNet(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)

        label2idx = [d.name for d in self.root.iterdir() if d.is_dir()]
        label2idx.sort()
        label2idx = {l: i for i, l in enumerate(label2idx)}

        self.labels = []
        length = 0
        for subdir in self.root.iterdir():
            if subdir.is_file():
                continue

            for _ in (subdir / "images").iterdir():
                length += 1

            self.labels.append((label2idx[subdir.name], subdir, length))

        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        prev_len = 0
        subdir = Path(".")
        label = -1
        for label, subdir, length in self.labels:
            if length > idx:
                idx = idx - prev_len
                break
            prev_len = length

        file_name = f"{subdir.name}_{idx}"
        tensor_path = subdir / "tensors" / f"{file_name}.npy"
        if not tensor_path.exists():
            tensor_path.parent.mkdir(exist_ok=True)
            image_path = subdir / "images" / f"{file_name}.JPEG"
            tensor = preprocess_image(image_path)
            np.save(tensor_path, tensor, allow_pickle=False)
        else:
            tensor = np.load(tensor_path)

        return tensor, label


class TinyImageTestNet(Dataset):
    def __init__(self, root_dir):
        self.dir = Path(root_dir)

        with (self.dir / "val_annotations.txt").open() as f:
            labels = [line.split("\t")[1] for line in f.readlines()]
        label2idx = list(set(labels))
        label2idx.sort()
        label2idx = {l: i for i, l in enumerate(label2idx)}
        self.labels = [label2idx[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor_path = self.dir / "tensors" / f"val_{idx}.npy"
        if not tensor_path.exists():
            tensor_path.parent.mkdir(exist_ok=True)
            image_path = self.dir / "images" / f"val_{idx}.JPEG"
            tensor = preprocess_image(image_path)
            np.save(tensor_path, tensor, allow_pickle=False)
        else:
            tensor = np.load(tensor_path)

        return tensor, self.labels[idx]
