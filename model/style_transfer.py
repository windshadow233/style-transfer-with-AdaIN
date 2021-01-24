from torch.nn import functional as F
from .adaIN import *
from .vgg19 import *


class StyleTransformer(nn.Module):
    def __init__(self):
        super(StyleTransformer, self).__init__()
        self.vgg_encoder = VGG19(state_dict='pre-trained/vgg19.pth', feature_mode=True).eval().requires_grad_(False)
        self.adaIN = AdaIN()
        self.decoder = Decoder()

    @staticmethod
    def calculate_content_loss(input, target):
        return F.mse_loss(input, target)

    def calculate_style_loss(self, input, target):
        input_mu = torch.mean(input, dim=(2, 3), keepdim=True)
        target_mu = torch.mean(target, dim=(2, 3), keepdim=True)
        input_var = torch.std(input, dim=(2, 3), keepdim=True) + self.adaIN.eps
        target_var = torch.std(target, dim=(2, 3), keepdim=True) + self.adaIN.eps
        return F.mse_loss(input_mu, target_mu) + F.mse_loss(input_var.sqrt(), target_var.sqrt())

    def forward(self, content, style, return_loss=False):
        content_feature_map = self.vgg_encoder(content)[-1]
        style_feature_maps = self.vgg_encoder(style)
        t = self.adaIN(content_feature_map, style_feature_maps[-1])
        result = self.decoder(t)
        if not return_loss:
            return result
        result_feature_maps = self.vgg_encoder(result)
        loss_c = self.calculate_content_loss(result_feature_maps[-1], t)
        loss_s = 0
        for result_feature, style_feature in zip(result_feature_maps, style_feature_maps):
            loss_s += self.calculate_style_loss(result_feature, style_feature)
        return loss_c, loss_s
