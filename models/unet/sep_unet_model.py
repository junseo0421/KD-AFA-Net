""" Full assembly of the parts to form the complete network """
import torch.nn

from models.unet.sep_unet_parts import *


class AFA_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AFA_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 32))  # first two separable conv -> regular conv
        self.down1 = (SepDown(32, 64))
        self.down2 = (SepDown(64, 128))
        self.down3 = (SepDown(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(256, 512 // factor))

        # decoder
        self.up1 = (SepUp(512, 256 // factor, bilinear))
        self.up2 = (SepUp(256, 128 // factor, bilinear))
        self.up3 = (SepUp(128, 64 // factor, bilinear))
        self.up4 = (SepUp(64, 32, bilinear))

        # Output layer
        self.outc = (OutConv(32, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.randn(256, 256, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.randn(256, 256, 1, 1))

        self.concat_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")
        x_shift = torch.fft.fftshift(x_ft)

        magnitude = torch.abs(x_shift)
        phase = torch.angle(x_shift)

        h, w = x_shift.shape[2:4]
        cy, cx = int(h / 2), int(w / 2)
        rh, rw = int(cuton * cy), int(cuton * cx)

        low_pass = torch.zeros_like(magnitude)
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att

        real = mag_out * torch.cos(phase)
        imag = mag_out * torch.sin(phase)

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out)

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real

        return out

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)
        x4_output = self.concat_conv4(x4_concat)
        x4_output = self.bn4(x4_output)
        x4_output = self.relu(x4_output)

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)
        x3_output = self.concat_conv3(x3_concat)
        x3_output = self.bn3(x3_output)
        x3_output = self.relu(x3_output)

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)
        x2_output = self.concat_conv2(x2_concat)
        x2_output = self.bn2(x2_output)
        x2_output = self.relu(x2_output)

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)
        x1_output = self.concat_conv1(x1_concat)
        x1_output = self.bn1(x1_output)
        x1_output = self.relu(x1_output)

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class LAFA_Net(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(LAFA_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))

        factor = 2 if bilinear else 1

        # decoder
        self.up1 = (SepUp(32, 16 // factor, bilinear))
        self.up2 = (SepUp(16, 8 // factor, bilinear))
        self.up3 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))

        self.concat_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")
        x_shift = torch.fft.fftshift(x_ft)

        magnitude = torch.abs(x_shift)
        phase = torch.angle(x_shift)

        h, w = x_shift.shape[2:4]
        cy, cx = int(h / 2), int(w / 2)
        rh, rw = int(cuton * cy), int(cuton * cx)

        low_pass = torch.zeros_like(magnitude)
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att

        real = mag_out * torch.cos(phase)
        imag = mag_out * torch.sin(phase)

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out)

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real

        return out

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)
        x3_output = self.concat_conv3(x3_concat)
        x3_output = self.bn3(x3_output)
        x3_output = self.relu(x3_output)

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)
        x2_output = self.concat_conv2(x2_concat)
        x2_output = self.bn2(x2_output)
        x2_output = self.relu(x2_output)

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)
        x1_output = self.concat_conv1(x1_concat)
        x1_output = self.bn1(x1_output)
        x1_output = self.relu(x1_output)

        # decoder
        x = self.up1(x4, x3_output)
        x = self.up2(x, x2_output)
        x = self.up3(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4}

