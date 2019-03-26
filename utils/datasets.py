import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys

import cv2

from data import TestBaseTransform


class ImageFolder(Dataset):

    def __init__(self, folder_path, shrink):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.shrink = shrink
        self.transform = TestBaseTransform((104, 117, 123))


    def __getitem__(self, index):
        # Extract image
        img_path = self.files[index % len(self.files)]
        img = cv2.imread(img_path)
        img_og = img

        if self.shrink != 1:
            img = cv2.resize(img, None, None, fx=self.shrink, fy=self.shrink, interpolation=cv2.INTER_LINEAR)

        img = self.transform(img)[0]

        # As pytorch tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img_og, img


    def __len__(self):
        return len(self.files)
