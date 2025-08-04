import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import math


class SSIM_loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_loss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x)
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM score
        return torch.mean(1-torch.clamp((SSIM_n / SSIM_d) / 2, 0, 1))



class Proj_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Proj_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projector = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, fm):
        modified_fm = self.projector(fm)
        return modified_fm


class Sobel_loss(nn.Module):
    def __init__(self):
        super(Sobel_loss, self).__init__()
        # Define Sobel kernels
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1],
                                                  [-2, 0, 2],
                                                  [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                    requires_grad=False)  # (1, 1, 3, 3)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                    requires_grad=False)  # (1, 1, 3, 3)

    def sobel_apply(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        # Compute gradient magnitude

        epsilon = 1e-8
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)

        return grad_magnitude

    def forward(self, pred, gt):
        if pred.shape[1] == 3:
            pred_gray = TF.rgb_to_grayscale(pred, num_output_channels=1)  # (N, 1, H, W)
        else:
            pred_gray = pred

        if gt.shape[1] == 3:
            gt_gray = TF.rgb_to_grayscale(gt, num_output_channels=1)  # (N, 1, H, W)
        else:
            gt_gray = gt

        sobel_pred = self.sobel_apply(pred_gray)
        sobel_gt = self.sobel_apply(gt_gray)

        return F.l1_loss(sobel_pred, sobel_gt)
