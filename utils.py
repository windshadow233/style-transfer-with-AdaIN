import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from model.style_transfer import StyleTransformer
from data.dataset import ImageDataset


def visualize(imgs, results):
    concatenated = torch.cat((imgs.cpu(), results.cpu()), 0)
    concatenated = denorm(concatenated)
    imshow(torchvision.utils.make_grid(concatenated, nrow=len(imgs)))


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.close()


def denorm(img):
    return img.clip(min=0, max=1)


transform = torchvision.transforms.ToTensor()


@torch.no_grad()
def test_image(model, content_img_path, style_img_path):
    content = Image.open(content_img_path).convert('RGB')
    style = Image.open(os.path.join(style_img_path, style_img_path)).convert('RGB')
    content = transform(content)[None].cuda()
    style = transform(style)[None].cuda()
    result = model(content, style).cpu()
    imshow(content.cpu().squeeze())
    imshow(denorm(result.squeeze()))


@torch.no_grad()
def show(model, dataset, samples_num=4):
    content, style = dataset(samples_num)
    content, style = content.cuda(), style.cuda()
    result = model(content, style, return_loss=False)
    visualize(content, result)


model = StyleTransformer()
model.load_state_dict(torch.load('trained_model/model2.pkl'))
model.cuda().eval()
dataset = ImageDataset(r'F:\datasets\Miyazaki Hayao2photo', transform, 'test')
