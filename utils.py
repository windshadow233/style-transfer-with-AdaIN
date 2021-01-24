import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np


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
    style = Image.open(style_img_path).convert('RGB')
    content = transform(content)[None].cuda()
    style = transform(style)[None].cuda()
    result = model(content, style).cpu()
    imshow(denorm(result.squeeze()))
