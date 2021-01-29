import sys
from torchvision import transforms as T
from torch import optim
from model.style_transfer import StyleTransformer
from data.dataset import ImageDataset
from itertools import count
from utils import *
import argparse
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self.content_loss = []
        self.style_loss = []

    def adjust_learning_rate(self, iteration_count):
        """Imitating the original implementation"""
        lr = self.init_lr / (1.0 + self.lr_decay * iteration_count)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def show(self, samples_num=4):
        content, style = self.dataset(samples_num)
        if self.use_cuda:
            content, style = content.cuda(), style.cuda()
        result = self.model(content, style, return_loss=False)
        visualize(content, result)

    def train(self):
        for iteration in count(start=1):
            self.optim.zero_grad()
            content, style = self.dataset(self.batch_size)
            if self.use_cuda:
                content, style = content.cuda(), style.cuda()
            loss_c, loss_s = self.model(content, style, return_loss=True)
            loss = self.content_weight * loss_c + self.style_weight * loss_s
            loss.backward()
            self.content_loss.append(loss_c.item())
            self.style_loss.append(loss_s.item())
            self.optim.step()
            print(f'\rIteration: {iteration} | Content Loss: {loss_c.item()} | Style Loss: {loss_s.item()}',
                  file=sys.stdout, flush=True, end='')
            self.adjust_learning_rate(iteration)
            if not iteration % self.show_result_every:
                 self.show()
            if iteration == self.max_iteration:
                plt.plot(self.content_loss)
                plt.title('Content Loss')
                plt.savefig('content_loss.jpg')
                plt.plot(self.style_loss)
                plt.title('Style Loss')
                plt.savefig('style_loss.jpg')
                break


if __name__ == '__main__':
    args = argparse.ArgumentParser('Args')
    args.add_argument('--data_root', type=str, default=r'F:\datasets\cartoonization')
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--lr_decay', type=float, default=5e-5)
    args.add_argument('--content_weight', '-cw', type=float, default=1)
    args.add_argument('--style_weight', '-sw', type=float, default=1e-2)
    args.add_argument('--max_iteration', '-mi', type=int, default=100000)
    args.add_argument('--save_path', '-sp', type=str, default='trained_model/model.pkl')
    args, _ = args.parse_known_args()
    trainer = Trainer(data_root=args.data_root,
                      lr=args.lr, lr_decay=args.lr_decay, content_weight=args.content_weight,
                      style_weight=args.style_weight, show_result_every=100,
                      max_iteration=args.max_iteration)
    trainer.train()
    torch.save(trainer.model.state_dict(), args.save_path)
