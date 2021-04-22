# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self,
                 root,
                 train=True,
                 year=2018,
                 transform=None,
                 target_transform=None,
                 category='name',
                 loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root,
                                 f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path,
                                    train=is_train,
                                    transform=transform,
                                    download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path,
                                   train=is_train,
                                   transform=transform,
                                   download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path,
                              train=is_train,
                              year=2018,
                              category=args.inat_category,
                              transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path,
                              train=is_train,
                              year=2019,
                              category=args.inat_category,
                              transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'mini-IN':
        dataset = miniImageNet(args.data_path,
                               mode=args.data_mode,
                               transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size,
                                                            padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size,
                interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class miniImageNet(torch.utils.data.Dataset):
    def __init__(self, data_root, mode='train', transform=None):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        self.transform = transform
        data = np.load(os.path.join(data_root, 'mini-ImageNet',
                                    f'mini-imagenet-cache-{mode}.pkl'),
                       allow_pickle=True)
        self.image_data = data['image_data']
        self.class_dict = data['class_dict']
        self.classes = sorted(list(self.class_dict.keys()))
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

        if mode == 'train':
            self.nb_classes = 64
        elif mode == 'val':
            self.nb_classes = 16
        else:
            self.nb_classes = 20

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imindex, label = self.samples[index]
        image = Image.fromarray(
            self.image_data[imindex].astype('uint8')).convert('RGB')
        image = self.transform(image)
        label = np.long(label)
        return image, label

    def _make_dataset(self):
        instances = []
        for target_class in sorted(self.cls_to_idx.keys()):
            class_index = self.cls_to_idx[target_class]
            target_index = self.class_dict[target_class]
            instances.extend([(_ind, class_index) for _ind in target_index])
        return instances


def build_fs_dataset(args):
    from torchmeta.datasets import MiniImagenet
    from torchmeta.transforms import Categorical, ClassSplitter, Rotation
    from torchvision.transforms import Compose, Resize, ToTensor
    from torchmeta.utils.data import BatchMetaDataLoader

    dataset = MiniImagenet(
        args.test_data_path,
        # Number of ways
        num_classes_per_task=5,
        # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
        transform=Compose([
            Resize((224, 224)),
            ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]),
        # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
        target_transform=Categorical(num_classes=5),
        meta_test=True,
        download=True)
    dataset = ClassSplitter(dataset,
                            shuffle=True,
                            num_train_per_class=5,
                            num_test_per_class=15)
    dataloader = BatchMetaDataLoader(dataset, batch_size=1, num_workers=2)
    return dataloader
