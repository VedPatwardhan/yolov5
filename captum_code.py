import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


class Tensor2Scalar(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        self.model.train()
        x = self.model(x)
        x = torch.mean(x, dim=[1, 2, 3, 4])
        return x


def captum_attribution(model, im, path):
    print("File Path: ", path)
    folder_name = path.split('/')[-1][0:-4]
    print("Folder Name", folder_name)
    os.makedirs('./results/{}'.format(folder_name))
    tensor2scalar = Tensor2Scalar(model)

    integrated_gradients = IntegratedGradients(tensor2scalar)
    attributions_ig = integrated_gradients.attribute(im)
    print("Attributions Integrated Gradients", attributions_ig.shape, im.shape)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
    fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(im.squeeze().cpu().detach().numpy(), (1,2,0)),
                             ['original_image', 'heat_map'],
                             ['all', 'positive'],
                             cmap=default_cmap,
                             show_colorbar=True)
    fig.savefig('./results/{}/integrated_gradients.png'.format(folder_name))

    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(im, nt_type='smoothgrad_sq')
    print("Attributions Noise Tunnel", attributions_ig_nt.shape)
    fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(im.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)
    fig.savefig('./results/{}/noise_tunnel.png'.format(folder_name))

    rand_img_dist = torch.cat([im * 0, im * 1])
    gradient_shap = GradientShap(tensor2scalar)
    attributions_gs = gradient_shap.attribute(im, baselines=rand_img_dist)
    print("Attributions Gradient Shap", attributions_gs.shape)
    fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(im.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)
    fig.savefig('./results/{}/gradient_shap.png'.format(folder_name))

    occlusion = Occlusion(tensor2scalar)
    attributions_occ = occlusion.attribute(im, sliding_window_shapes=(3,15, 15))
    print("Attributions Occlusion", attributions_occ.shape)
    fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(im.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)
    fig.savefig('./results/{}/occlusion.png'.format(folder_name))
