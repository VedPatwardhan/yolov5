import numpy as np
import cv2

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel



class Tensor2Scalar(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        self.model.train()
        x = self.model(x)
        x = torch.mean(x, dim=[1, 2, 3, 4])
        return x


def captum_attribution(model, im):
    tensor2scalar = Tensor2Scalar(model)

    integrated_gradients = IntegratedGradients(tensor2scalar)
    attributions_ig = integrated_gradients.attribute(im)
    print("Attributions Integrated Gradients", attributions_ig.shape)
    cv2.imshow(np.transpose(attributions_ig.cpu().detach().numpy(), (1, 2, 0)))
    cv2.waitKey(10000)

    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(im, nt_type='smoothgrad_sq')
    print("Attributions Noise Tunnel", attributions_ig_nt.shape)

    rand_img_dist = torch.cat([im * 0, im * 1])
    gradient_shap = GradientShap(tensor2scalar)
    attributions_gs = gradient_shap.attribute(im, baselines=rand_img_dist)
    print("Attributions Gradient Shap", attributions_gs.shape)

    occlusion = Occlusion(tensor2scalar)
    attributions_occ = occlusion.attribute(im, sliding_window_shapes=(3,15, 15))
    print("Attributions Occlusion", attributions_occ.shape)
