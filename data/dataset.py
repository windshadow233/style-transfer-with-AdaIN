import glob
import os
from PIL import Image
import random
import torch
import numpy as np


class ImageDataset(object):
    def __init__(self, root=r'F:\datasets\Miyazaki Hayao2photo', transform=None, mode='train'):
        self.root = root
        self.mode = mode
        self.load_data('A')
        self.load_data('B')
        self.transform = transform

    def load_data(self, img_type):
        attr = {'A': 'style', 'B': 'content'}[img_type] + '_images'
        self.__setattr__(attr, glob.glob(os.path.join(self.root, self.mode + img_type, '*')))
        random.shuffle(self.__getattribute__(attr))

    def __call__(self, batch_size):
        if len(self.style_images) < batch_size:
            self.load_data(img_type='A')
        if len(self.content_images) < batch_size:
            self.load_data(img_type='B')
        content_images, self.content_images = self.content_images[:batch_size], self.content_images[batch_size:]
        style_images, self.style_images = self.style_images[:batch_size], self.style_images[batch_size:]
        if self.transform is not None:
            content_images = torch.stack([self.transform(Image.open(img_path).convert('RGB')) for img_path in content_images])
            style_images = torch.stack([self.transform(Image.open(img_path).convert('RGB')) for img_path in style_images])
        else:
            content_images = np.stack([np.array(Image.open(img_path).convert('RGB')) for img_path in self.content_images]).transpose((0, 3, 1, 2))
            style_images = np.stack([np.array(Image.open(img_path).convert('RGB')) for img_path in self.style_images]).transpose((0, 3, 1, 2))
        return content_images, style_images
