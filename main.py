import torch
import torchvision
import sys
from torchvision import transforms as T
from torch import optim
from model.style_transfer import StyleTransformer
from data.dataset import ImageDataset
from itertools import count
from matplotlib import pyplot as plt
import numpy as np


class Trainer(object):
    def __init__(self, data_root=r'F:\datasets\vangogh2photo', lr=2e-4, lr_decay=5e-5, content_weight=1,
                 style_weight=1e-2, use_cuda=True, show_result_every=100, max_iteration=160000):
        self.model = StyleTransformer()
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        self.dataset = ImageDataset(
            root=data_root,
            transform=T.Compose([
                # T.RandomCrop(size=(128, 128)),
                T.ToTensor(),
                # T.Normalize([0.5], [0.5])
            ]))
        self.init_lr = lr
        self.lr_decay = lr_decay
        self.optim = optim.Adam(self.model.decoder.parameters(), lr=lr)
        self.batch_size = 8
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.show_result_every = show_result_every
        self.max_iteration = max_iteration

    def adjust_learning_rate(self, iteration_count):
        """Imitating the original implementation"""
        lr = self.init_lr / (1.0 + self.lr_decay * iteration_count)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def visualize(self, imgs, results):
        concatenated = torch.cat((imgs.cpu(), results.cpu()), 0)
        concatenated = self.denorm(concatenated)
        self.imshow(torchvision.utils.make_grid(concatenated, nrow=len(imgs)))

    @staticmethod
    def imshow(img):
        npimg = img.numpy()
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        plt.close()

    @staticmethod
    def denorm(img):
        return img.clip(min=0, max=1)

    @torch.no_grad()
    def show(self, samples_num=4):
        content, style = self.dataset(samples_num)
        if self.use_cuda:
            content, style = content.cuda(), style.cuda()
        result = self.model(content, style, return_loss=False)
        self.visualize(content, result)

    def train(self):
        for iteration in count(start=1):
            self.optim.zero_grad()
            content, style = self.dataset(self.batch_size)
            if self.use_cuda:
                content, style = content.cuda(), style.cuda()
            loss_c, loss_s = self.model(content, style, return_loss=True)
            loss = self.content_weight * loss_c + self.style_weight * loss_s
            loss.backward()
            self.optim.step()
            print(f'\rIteration: {iteration} | Content Loss: {loss_c.item()} | Style Loss: {loss_s.item()}',
                  file=sys.stdout, flush=True, end='')
            self.adjust_learning_rate(iteration)
            if not iteration % self.show_result_every:
                self.show()
            if iteration == self.max_iteration:
                break


if __name__ == '__main__':
    trainer = Trainer(data_root=r'F:\datasets\monet2photo',
                      lr=1e-4, lr_decay=5e-5, content_weight=10,
                      style_weight=1e-1, show_result_every=50, max_iteration=20000)
    trainer.train()
