import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from skimage.transform import resize

import sys

import cv2



class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = cv2.imread(img_path)
        # h, w, _ = img.shape
        # dim_diff = np.abs(h - w)
        # # Upper (left) and lower (right) padding
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # # Determine padding
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # # Add padding
        # input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # # Resize and normalize
        # input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # # Channels-first
        # input_img = np.transpose(input_img, (2, 0, 1))
        # # As pytorch tensor
        # input_img = torch.from_numpy(input_img).float()

        return img_path, img

    def __len__(self):
        return len(self.files)
