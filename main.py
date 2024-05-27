"""
2023. 04. 18.

Code for generating 3-component ground motion data using the StyleGAN2 model

This code was created with reference to the following repositories:
    ・https://github.com/NVlabs/stylegan2/tree/master
    ・https://github.com/ayukat1016/gan_sample.git
    ・https://github.com/rosinality/id-gan-pytorch/tree/master/stylegan2
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()

# ----------------------------------------------------------------------------------------------------------------------
# --------- parameter settings ----------------
csv_path = "../data/input_file.csv"
out_dir = "../data/out"

input_label_list = ["mw", "log_fault_dist", "log_v30", "log_z1500", "log10_pga"]

w_dim = 512
z_dim = 512

num_epoch = 100000
batch_size = 64

gen_regularization_interval = 4
disc_regularization_interval = 16
gen_train_num = 4
disc_train_num = 4

base_lr = 0.002
base_beta1 = 0.0
base_beta2 = 0.99

# ------------------------------------ do not change -------------------------------------------------------------------
label_num = len(input_label_list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Adam parameters
def get_adam_params_adjust_interval(reg_interval, learning_rate, beta_1_a, beta_2_a):
    mini_batch_ratio = reg_interval / (reg_interval + 1)
    l_rate = learning_rate * mini_batch_ratio
    b1 = beta_1_a**mini_batch_ratio
    b2 = beta_2_a**mini_batch_ratio
    return l_rate, b1, b2


g_lr, g_beta1, g_beta2 = get_adam_params_adjust_interval(gen_regularization_interval, base_lr, base_beta1, base_beta2)
d_lr, d_beta1, d_beta2 = get_adam_params_adjust_interval(disc_regularization_interval, base_lr, base_beta1, base_beta2)
print(g_lr, g_beta1, g_beta2)
print(d_lr, d_beta1, d_beta2)

rng = np.random.RandomState(1234)
torch.manual_seed(1234)


# ----------------------------------------------------------------------------------------------------------------------
# Mapping networkに入力する前に潜在変数を正規化する
def noise_normalization(input_noise):
    noise_var = torch.mean(input_noise**2, dim=1, keepdim=True)

    return input_noise / torch.sqrt(noise_var + 1e-8)


class GroundMotionDatasets(Dataset):
    def __init__(self, csv_path_class, input_label_list_class):
        df = pd.read_csv(csv_path_class)
        self.data_path = df["file_name"]
        self.label = torch.from_numpy(df[input_label_list_class].values.astype(np.float32)).clone()
        # print(self.label.size())

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        temp_mat = np.load(path, allow_pickle=True)
        out_tensor = torch.from_numpy(temp_mat.astype(np.float32)).clone()
        out_tensor = out_tensor.T.reshape(3, 1, -1)

        out_label = self.label[index, :]

        return out_tensor, out_label


# Amplify the signal of feature map.
class Amplify(nn.Module):
    def __init__(self, rate):
        super(Amplify, self).__init__()
        self.rate = rate

    def forward(self, x):
        return x * self.rate


# Add bias to the input tensor.
class AddBiasChannelWise(nn.Module):
    def __init__(self, out_channels, bias_scale):
        super(AddBiasChannelWise, self).__init__()

        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scale = bias_scale

    def forward(self, x):
        bias_len, *_ = self.bias.shape
        # print(bias_len)
        new_shape = (1, bias_len) if x.ndim == 2 else (1, bias_len, 1, 1)
        # print(new_shape)
        # print(x.shape)
        y = x + self.bias.view(*new_shape) * self.bias_scale

        return y


# Dense layer with equalized learning rate and custom learning rate multiplier.
class EqualizedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn((output_dim, input_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / learning_rate)
        self.weight_scale = 1.0 / (input_dim**0.5) * learning_rate

    def forward(self, x):
        return functional.linear(x, weight=self.weight * self.weight_scale, bias=None)


# -------------------------- Mapping network -----------------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dimension, w_dimension, start_size=16, end_size=4096):
        super(MappingNetwork, self).__init__()

        self.style_num = int(np.log2(end_size / start_size)) * 2 + 2

        self.model = nn.Sequential(
            # 1
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 2
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 3
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 4
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 5
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 6
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 7
            EqualizedLinear(input_dim=z_dimension, output_dim=z_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=z_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 8
            EqualizedLinear(input_dim=z_dimension, output_dim=w_dimension, learning_rate=0.01),
            AddBiasChannelWise(out_channels=w_dimension, bias_scale=0.01),
            Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = noise_normalization(x)
        x = self.model(x)
        # print(x.size())
        batch_size_c, vector_len_c = x.shape
        x = x.view(batch_size_c, 1, vector_len_c).expand(batch_size_c, self.style_num, vector_len_c)

        return x


class ModulateConv2d(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size_h,
        kernel_size_w,
        padding_num,
        stride_width,
        w_dimension,
        lr_mul=1.0,
        demodulate=True,
    ):
        super(ModulateConv2d, self).__init__()

        self.stride_width = stride_width
        self.padding_num = padding_num
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size_h, kernel_size_w))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / lr_mul)
        self.weight_scale = 1 / np.sqrt(input_channels * kernel_size_h * kernel_size_w) * lr_mul

        self.linear = EqualizedLinear(input_dim=w_dimension, output_dim=input_channels, learning_rate=lr_mul)
        self.bias = AddBiasChannelWise(out_channels=input_channels, bias_scale=lr_mul)

    def forward(self, pack):
        x, style = pack
        batch_size_c, channel_num_c, height_c, width_c = x.shape
        out_channel_num, in_channel_num, kernel_height, kernel_width = self.weight.shape

        # [b, 512] -> [b, 512]
        temp_mod_style = self.linear(style)
        mod_style = self.bias(temp_mod_style) + 1

        # [b, input_channels] -> [b, 1, input_channels, 1, 1]
        mod_style = mod_style.view(batch_size_c, 1, in_channel_num, 1, 1)

        # [out, in, kernel, kernel] -> [1, out, in, kernel, kernel]
        resized_weight = self.weight.view(1, out_channel_num, in_channel_num, kernel_height, kernel_width)

        # weight          : 1,     out_c, in_c, kh, kw
        # style           : batch,     1, in_c,  1,  1
        # modulated_weight: batch, out_c, in_c, kh, kw
        modulated_weight = resized_weight * mod_style * self.weight_scale
        # print(modulated_weight.size())

        if self.demodulate:
            weight_sum = modulated_weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8
            demodulate_norm = torch.rsqrt(weight_sum).view(batch_size_c, out_channel_num, 1, 1, 1)
            weight_ = modulated_weight * demodulate_norm
        else:
            weight_ = modulated_weight

        # 畳み込み計算をする
        # Input: [1, b*channel, H, W]
        # Weight: [b*out_c, in_c, kH, kW]
        # print(weight_.shape)
        weight = weight_.view(batch_size_c * out_channel_num, in_channel_num, kernel_height, kernel_width)
        x = x.view(1, batch_size_c * channel_num_c, height_c, width_c)

        out = functional.conv2d(
            x, weight=weight, padding=(0, self.padding_num), stride=self.stride_width, groups=batch_size_c
        )
        # print(out.shape)
        out = out.view(batch_size_c, out_channel_num, height_c, width_c)

        return out


class UpSampleConv2d(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size_h,
        kernel_size_w,
        padding_num,
        stride_width,
        w_dimension,
        lr_mul=1.0,
        demodulate=True,
    ):
        super(UpSampleConv2d, self).__init__()

        self.stride_width = stride_width
        self.padding_num = padding_num
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(input_channels, output_channels, kernel_size_h, kernel_size_w))
        torch.nn.init.normal_(self.weight.data, mean=1.0, std=1.0 / lr_mul)
        self.weight_scale = 1.0 / np.sqrt(input_channels * kernel_size_h * kernel_size_w) * lr_mul

        self.linear = EqualizedLinear(input_dim=w_dimension, output_dim=input_channels, learning_rate=lr_mul)
        self.bias = AddBiasChannelWise(out_channels=input_channels, bias_scale=lr_mul)

    def forward(self, pack):
        x, w = pack
        batch_size_c, channel_num_c, height_c, width_c = x.shape
        in_channel_num, out_channel_num, kernel_height, kernel_width = self.weight.shape

        # [b, 512] -> [b, 512]
        temp_mod_style = self.linear(w)
        mod_style = self.bias(temp_mod_style) + 1

        # apply style
        mod_style = mod_style.view(batch_size_c, in_channel_num, 1, 1, 1)
        resized_weight = self.weight.view(1, in_channel_num, out_channel_num, kernel_height, kernel_width)
        modulated_weight = resized_weight * mod_style * self.weight_scale

        if self.demodulate:
            weight_sum = modulated_weight.pow(2).sum(dim=[1, 3, 4]) + 1e-8
            demodulate_norm = torch.rsqrt(weight_sum).view(batch_size_c, 1, out_channel_num, 1, 1)
            weight_ = modulated_weight * demodulate_norm
        else:
            weight_ = modulated_weight

        weight = weight_.view(batch_size_c * in_channel_num, out_channel_num, kernel_height, kernel_width)
        x = x.view(1, batch_size_c * channel_num_c, height_c, width_c)
        # print(x.shape)
        # print(weight.shape)

        out = functional.conv_transpose2d(
            x, weight=weight, padding=self.padding_num, stride=self.stride_width, groups=batch_size_c
        )

        _, _, temp_h, temp_w = out.shape
        out = out.view(batch_size_c, out_channel_num, temp_h, temp_w)

        return out


class BlurPooling(nn.Module):
    def __init__(self, input_channels):
        super(BlurPooling, self).__init__()

        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(input_channels, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)
        # print(self.const_blur_filter[0])
        # print(self.const_blur_filter.shape)

    def forward(self, x):
        batch_size_c, channel_num_c, height_c, width_c = x.shape
        x = functional.pad(x, (1, 1), mode="constant", value=0)
        x = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=channel_num_c)

        return x


class UpSampleWave(nn.Module):
    def __init__(self):
        super(UpSampleWave, self).__init__()

        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(3, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)

    def forward(self, x):
        batch_size_c, channel_num_c, height_c, width_c = x.shape

        # # [b, 1, 1, n] -> [b, 1, n, 1]
        x = x.reshape(batch_size_c, channel_num_c, width_c, height_c)

        x = functional.pad(x, (0, 1), mode="constant", value=0)

        # # [b, 1, n, 2] -> [b, 1, 1, n*2]
        x = x.reshape(batch_size_c, channel_num_c, height_c, -1)

        x = functional.pad(x, (2, 1), mode="constant", value=0)

        x = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=channel_num_c)

        return x


class PixelWiseNoise(nn.Module):
    def __init__(self, wave_width):
        super(PixelWiseNoise, self).__init__()
        self.register_buffer("const_noise", torch.randn(1, 1, 1, wave_width))
        self.noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_c, channel_c, width_h, width_c = x.shape
        noise = self.const_noise.expand(batch_c, channel_c, width_h, width_c)

        return x + noise * self.noise_scale


class PgaPrediction(nn.Module):
    def __init__(self, in_channels, label_number):
        super(PgaPrediction, self).__init__()

        self.model = nn.Sequential(
            # 1
            EqualizedLinear(input_dim=in_channels, output_dim=4096 * 2, learning_rate=1),
            AddBiasChannelWise(out_channels=4096 * 2, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 2
            EqualizedLinear(input_dim=4096 * 2, output_dim=2048, learning_rate=1),
            AddBiasChannelWise(out_channels=2048, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 3
            EqualizedLinear(input_dim=2048, output_dim=512, learning_rate=1),
            AddBiasChannelWise(out_channels=512, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 4
            EqualizedLinear(input_dim=512, output_dim=128, learning_rate=1),
            AddBiasChannelWise(out_channels=128, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 5
            EqualizedLinear(input_dim=128, output_dim=32, learning_rate=1),
            AddBiasChannelWise(out_channels=32, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 6
            EqualizedLinear(input_dim=32, output_dim=16, learning_rate=1),
            AddBiasChannelWise(out_channels=16, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 7
            EqualizedLinear(input_dim=16, output_dim=8, learning_rate=1),
            AddBiasChannelWise(out_channels=8, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 8
            EqualizedLinear(input_dim=8, output_dim=label_number, learning_rate=1),
            AddBiasChannelWise(out_channels=label_number, bias_scale=1),
        )

    def forward(self, z):
        pre_pga = self.model(z)

        return pre_pga


class SynthesisNetwork(nn.Module):
    def __init__(self, w_dimension, label_number):
        super(SynthesisNetwork, self).__init__()

        self.const_input = nn.Parameter(torch.randn(1, 512, 1, 16))
        self.pga_prediction = PgaPrediction(in_channels=8 * 4096, label_number=label_number)

        self.normal_conv_16 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=1,
                stride_width=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=16),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_16_to_32 = nn.Sequential(
            UpSampleConv2d(
                input_channels=512,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=512),
            PixelWiseNoise(wave_width=32),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_32 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=32),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_32_to_64 = nn.Sequential(
            UpSampleConv2d(
                input_channels=512,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=512),
            PixelWiseNoise(wave_width=64),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_64 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=64),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_64_to_128 = nn.Sequential(
            UpSampleConv2d(
                input_channels=512,
                output_channels=256,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=256),
            PixelWiseNoise(wave_width=128),
            AddBiasChannelWise(out_channels=256, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_128 = nn.Sequential(
            ModulateConv2d(
                input_channels=256,
                output_channels=256,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=128),
            AddBiasChannelWise(out_channels=256, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_128_to_256 = nn.Sequential(
            UpSampleConv2d(
                input_channels=256,
                output_channels=128,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=128),
            PixelWiseNoise(wave_width=256),
            AddBiasChannelWise(out_channels=128, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_256 = nn.Sequential(
            ModulateConv2d(
                input_channels=128,
                output_channels=128,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=256),
            AddBiasChannelWise(out_channels=128, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_256_to_512 = nn.Sequential(
            UpSampleConv2d(
                input_channels=128,
                output_channels=64,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=64),
            PixelWiseNoise(wave_width=512),
            AddBiasChannelWise(out_channels=64, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_512 = nn.Sequential(
            ModulateConv2d(
                input_channels=64,
                output_channels=64,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=512),
            AddBiasChannelWise(out_channels=64, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_512_to_1024 = nn.Sequential(
            UpSampleConv2d(
                input_channels=64,
                output_channels=32,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=32),
            PixelWiseNoise(wave_width=1024),
            AddBiasChannelWise(out_channels=32, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_1024 = nn.Sequential(
            ModulateConv2d(
                input_channels=32,
                output_channels=32,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=1024),
            AddBiasChannelWise(out_channels=32, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_1024_to_2048 = nn.Sequential(
            UpSampleConv2d(
                input_channels=32,
                output_channels=16,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=16),
            PixelWiseNoise(wave_width=2048),
            AddBiasChannelWise(out_channels=16, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_2048 = nn.Sequential(
            ModulateConv2d(
                input_channels=16,
                output_channels=16,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=2048),
            AddBiasChannelWise(out_channels=16, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.up_sample_conv_2048_to_4096 = nn.Sequential(
            UpSampleConv2d(
                input_channels=16,
                output_channels=8,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=0,
                stride_width=2,
                w_dimension=w_dimension,
                lr_mul=1.0,
                demodulate=True,
            ),
            BlurPooling(input_channels=8),
            PixelWiseNoise(wave_width=4096),
            AddBiasChannelWise(out_channels=8, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_4096 = nn.Sequential(
            ModulateConv2d(
                input_channels=8,
                output_channels=8,
                kernel_size_h=1,
                kernel_size_w=3,
                stride_width=1,
                padding_num=1,
                w_dimension=w_dimension,
            ),
            PixelWiseNoise(wave_width=4096),
            AddBiasChannelWise(out_channels=8, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.to_wave_16 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_32 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_64 = nn.Sequential(
            ModulateConv2d(
                input_channels=512,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_128 = nn.Sequential(
            ModulateConv2d(
                input_channels=256,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_256 = nn.Sequential(
            ModulateConv2d(
                input_channels=128,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_512 = nn.Sequential(
            ModulateConv2d(
                input_channels=64,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_1024 = nn.Sequential(
            ModulateConv2d(
                input_channels=32,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_2048 = nn.Sequential(
            ModulateConv2d(
                input_channels=16,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.to_wave_4096 = nn.Sequential(
            ModulateConv2d(
                input_channels=8,
                output_channels=3,
                kernel_size_h=1,
                kernel_size_w=1,
                stride_width=1,
                padding_num=0,
                w_dimension=w_dimension,
                demodulate=False,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.main_net_list = [
            self.up_sample_conv_16_to_32,
            self.normal_conv_32,
            self.up_sample_conv_32_to_64,
            self.normal_conv_64,
            self.up_sample_conv_64_to_128,
            self.normal_conv_128,
            self.up_sample_conv_128_to_256,
            self.normal_conv_256,
            self.up_sample_conv_256_to_512,
            self.normal_conv_512,
            self.up_sample_conv_512_to_1024,
            self.normal_conv_1024,
            self.up_sample_conv_1024_to_2048,
            self.normal_conv_2048,
            self.up_sample_conv_2048_to_4096,
            self.normal_conv_4096,
        ]
        self.to_wave_list = [
            self.to_wave_32,
            self.to_wave_64,
            self.to_wave_128,
            self.to_wave_256,
            self.to_wave_512,
            self.to_wave_1024,
            self.to_wave_2048,
            self.to_wave_4096,
        ]

        self.up_sample_wave = UpSampleWave()

    def forward(self, w):
        batch_num, style_num, w_feature = w.shape

        b_const_input = self.const_input.repeat(batch_num, 1, 1, 1)

        f_map = self.normal_conv_16([b_const_input, w[:, 0]])
        temp_wave = self.to_wave_16([f_map, w[:, 1]])
        # skip_wave = None

        # print(f_map.shape)
        # print(temp_wave.shape)
        # print(skip_wave.shape)

        for i in range(len(self.to_wave_list)):
            f_map = self.main_net_list[i * 2]([f_map, w[:, i * 2 + 1]])
            # print(f_map.shape)
            f_map = self.main_net_list[i * 2 + 1]([f_map, w[:, i * 2 + 2]])
            # print(f_map.shape)
            skip_wave = self.up_sample_wave(temp_wave)
            # print(skip_wave.shape)
            temp_wave = self.to_wave_list[i]([f_map, w[:, i * 2 + 3]]) + skip_wave

        pga = self.pga_prediction(f_map.reshape(-1, 4096 * 8))

        return temp_wave, pga


class Generator(nn.Module):
    def __init__(self, z_dimension, w_dimension, label_number, style_mixing_prob=0.9):
        super(Generator, self).__init__()

        self.mapping_network = MappingNetwork(z_dimension=z_dimension, w_dimension=w_dimension)
        self.synthesis_network = SynthesisNetwork(w_dimension=w_dimension, label_number=label_number)
        self.style_mixing_prob = style_mixing_prob

    def forward(self, z, is_train=True):
        s1 = self.mapping_network(z)
        s = s1

        if is_train:
            temp = np.random.uniform(low=0, high=1)
            if temp < self.style_mixing_prob:
                z2 = torch.randn(size=(s1.shape[0], z_dim)).to(device)
                s2 = self.mapping_network(z2)
                mix_index = np.random.randint(low=0, high=18)

                s = torch.cat([s1[:, :mix_index, :], s2[:, mix_index:, :]], dim=1)
            else:
                pass
        else:
            pass

        x, pga = self.synthesis_network(s)
        return x, s, pga


# ----------------------------------------------------------------------------------------------------------------------
# ------------------ Discriminator ------------------------------
# --

class Conv2dLayer(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size_h, kernel_size_w, padding_num, stride_width, lr_mul
    ):
        super(Conv2dLayer, self).__init__()

        self.stride_width = stride_width
        self.padding_num = padding_num

        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size_h, kernel_size_w))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / lr_mul)
        self.weight_scale = 1 / np.sqrt(input_channels * kernel_size_h * kernel_size_w) * lr_mul

    def forward(self, x):
        x = functional.conv2d(
            x, weight=self.weight * self.weight_scale, padding=(0, self.padding_num), stride=self.stride_width
        )
        return x


class FromWave(nn.Module):
    def __init__(self, input_channels, output_channels, lr_mul):
        super(FromWave, self).__init__()
        self.conv2d = Conv2dLayer(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size_h=1,
            kernel_size_w=1,
            padding_num=0,
            stride_width=1,
            lr_mul=1.0,
        )
        self.bias = AddBiasChannelWise(out_channels=output_channels, bias_scale=lr_mul)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.conv2d(x)
        out = out + self.bias(out)
        out = self.activation(out)
        return out


class DownSampleConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size_w):
        super(DownSampleConv2d, self).__init__()

        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, 1, kernel_size_w))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0)
        self.weight_scale = 1 / np.sqrt(input_channels * 1 * kernel_size_w) * 1.0

    def forward(self, x):
        x = functional.conv2d(input=x, weight=self.weight * self.weight_scale, padding=(0, 0), stride=(1, 2))

        return x


class BlurPoolDiscriminator(nn.Module):
    def __init__(self, input_channels, pad_num):
        super(BlurPoolDiscriminator, self).__init__()

        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(input_channels, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)

        self.pad_num = pad_num

    def forward(self, x):
        batch_size_c, channel_num_c, height_c, width_c = x.shape
        x = functional.pad(x, (self.pad_num, self.pad_num), mode="constant", value=0)

        x = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=channel_num_c)

        return x


class ResBlockDiscriminator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResBlockDiscriminator, self).__init__()

        self.main_model = nn.Sequential(
            Conv2dLayer(
                input_channels=input_channels,
                output_channels=input_channels,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=1,
                stride_width=1,
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=input_channels, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
            BlurPoolDiscriminator(input_channels=input_channels, pad_num=2),
            DownSampleConv2d(input_channels=input_channels, output_channels=output_channels, kernel_size_w=3),
            AddBiasChannelWise(out_channels=output_channels, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.blur_pool_skip = BlurPoolDiscriminator(input_channels=input_channels, pad_num=1)
        self.down_sample_conv_skip = DownSampleConv2d(
            input_channels=input_channels, output_channels=output_channels, kernel_size_w=1
        )

    def forward(self, x):
        skip = x
        x = self.main_model(x)
        skip = self.blur_pool_skip(skip)
        skip = self.down_sample_conv_skip(skip)
        # print(x.shape)
        # print(skip.shape)

        x = (x + skip) * (1.0 / np.sqrt(2))

        return x


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size, num_features):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x):
        # x: [b, 512, 1, 4]
        my_group_size = min(self.group_size, x.shape[0])
        batch_size_d, channel_d, height_d, width_d = x.shape

        # y: [group_size, b/group_size, num_features, 512/num_features, 1, 4]
        y = x.view((my_group_size, -1, self.num_features, channel_d // self.num_features, height_d, width_d))

        # y.mean: [1, 8, 1, 512, 1, 4]
        y = y - y.mean(dim=0, keepdim=True)

        y = y * y
        y = y.mean(dim=0)
        y = torch.sqrt(y + 1e-8)

        y = y.mean(dim=[2, 3, 4], keepdim=True)

        y = y.mean(dim=2)

        y = y.repeat([my_group_size, 1, height_d, width_d])

        x = torch.cat([x, y], dim=1)

        return x


class Discriminator(nn.Module):
    def __init__(self, label_number):
        super(Discriminator, self).__init__()

        self.from_wave = FromWave(input_channels=3, output_channels=32, lr_mul=1.0)

        self.res_block_1 = ResBlockDiscriminator(input_channels=32, output_channels=64)  # 1x4096 -> 1x2048
        self.res_block_2 = ResBlockDiscriminator(input_channels=64, output_channels=128)  # 1x2048 -> 1x1024
        self.res_block_3 = ResBlockDiscriminator(input_channels=128, output_channels=256)  # 1x1024 -> 1x512
        self.res_block_4 = ResBlockDiscriminator(input_channels=256, output_channels=512)  # 1x512  -> 1x256
        self.res_block_5 = ResBlockDiscriminator(input_channels=512, output_channels=512)  # 1x256  -> 1x128
        self.res_block_6 = ResBlockDiscriminator(input_channels=512, output_channels=512)  # 1x128  -> 1x64
        self.res_block_7 = ResBlockDiscriminator(input_channels=512, output_channels=512)  # 1x64   -> 1x32
        self.res_block_8 = ResBlockDiscriminator(input_channels=512, output_channels=512)  # 1x32   -> 1x16

        self.model_final = nn.Sequential(
            MiniBatchStdDev(group_size=4, num_features=1),
            Conv2dLayer(
                input_channels=513,
                output_channels=512,
                kernel_size_h=1,
                kernel_size_w=3,
                padding_num=1,
                stride_width=(1, 1),
                lr_mul=1.0,
            ),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.model_final_dense = nn.Sequential(
            EqualizedLinear(input_dim=512 * 16, output_dim=512, learning_rate=1.0),
            AddBiasChannelWise(out_channels=512, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
            EqualizedLinear(input_dim=512, output_dim=1, learning_rate=1.0),
            AddBiasChannelWise(out_channels=1, bias_scale=1.0),
        )

        self.embedding = nn.Sequential(
            # 1
            EqualizedLinear(input_dim=label_number, output_dim=128, learning_rate=1.0),
            AddBiasChannelWise(out_channels=128, bias_scale=1.0),
            nn.LeakyReLU(negative_slope=0.2),
            # 2
            EqualizedLinear(input_dim=128, output_dim=1024, learning_rate=1.0),
            AddBiasChannelWise(out_channels=1024, bias_scale=1.0),
            nn.LeakyReLU(negative_slope=0.2),
            # 3
            EqualizedLinear(input_dim=1024, output_dim=512 * 16, learning_rate=1.0),
            AddBiasChannelWise(out_channels=512 * 16, bias_scale=1.0),
        )

    def forward(self, x, y):
        batch_c = x.shape[0]
        x = self.from_wave(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.res_block_6(x)
        x = self.res_block_7(x)
        x = self.res_block_8(x)
        x = self.model_final(x)
        x = x.reshape(batch_c, -1)
        emb_y = self.embedding(y)
        out_y = torch.sum(x * emb_y, dim=1, keepdim=True)
        x = self.model_final_dense(x) + out_y

        return x


# # # Training ------------------------------------------------------------------------------
# requires_grad
def set_model_requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def generator_loss(disc_fake_wave):
    return torch.mean(functional.softplus(-disc_fake_wave))


def discriminator_loss(disc_fake_wave, disc_real_wave):
    loss_f = torch.mean(functional.softplus(disc_fake_wave)) + torch.mean(functional.softplus(-disc_real_wave))
    return loss_f


class GeneratorLossPathRegularization(nn.Module):
    def __init__(self, pl_decay=0.01, pl_weight=2.0, g_reg_int=4):
        super(GeneratorLossPathRegularization, self).__init__()

        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean_var = torch.zeros((1,)).to(device)
        self.reg_interval = g_reg_int

    def forward(self, fake_wave_f, fake_style_f):
        pl_noise = torch.randn(fake_wave_f.shape) / np.sqrt(np.prod(fake_style_f.shape[2:]))
        pl_noise = pl_noise.to(device)
        f_img_out_pl_n = torch.sum(fake_wave_f * pl_noise)
        pl_grads = torch.autograd.grad(outputs=f_img_out_pl_n, inputs=fake_style_f, create_graph=True)[0]
        pl_grads_sum_mean = pl_grads.pow(2).sum(dim=2).mean(dim=1)
        pl_length = torch.sqrt(pl_grads_sum_mean)

        # Track exponential moving average of |J*y|.
        pl_mean = self.pl_mean_var + self.pl_decay * (pl_length.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach()

        # Calculate (|J*y|-a)^2.
        pl_penalty = (pl_length - pl_mean).pow(2).mean()
        reg = pl_penalty * self.pl_weight * self.reg_interval
        return reg, pl_length.mean()


def discriminator_loss_r1(disc_real_wave, reals_f, gamma_f=10, d_reg_int=16):
    real_grads = torch.autograd.grad(outputs=torch.sum(disc_real_wave), inputs=reals_f, create_graph=True)[0]
    gradient_penalty = torch.sum(real_grads**2, dim=[1, 2, 3])
    reg = (gradient_penalty * gamma_f * 0.5 * d_reg_int).mean()
    return reg


if __name__ == "__main__":
    # Initialize
    netG = Generator(z_dimension=z_dim, w_dimension=w_dim, label_number=label_num).to(device)
    netD = Discriminator(label_number=label_num).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(g_beta1, g_beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(d_beta1, d_beta2))

    # load the data
    wave_dataset = GroundMotionDatasets(csv_path_class=csv_path, input_label_list_class=input_label_list)
    data_loader = DataLoader(wave_dataset, batch_size=batch_size, shuffle=True)

    time = np.linspace(0.0, 40.95, 4096)

    netG.train()
    netD.train()

    g_loss_list = []
    g_loss_reg_list = []
    g_train_prob_list = []
    d_train_prob_list_real = []
    d_train_prob_list_fake = []
    d_loss_list = []
    d_loss_reg_list = []

    fixed_noise = torch.randn(10, z_dim).to(device)

    loss_func_g_reg = GeneratorLossPathRegularization(g_reg_int=gen_regularization_interval).to(device)

    for epoch in range(num_epoch):
        # Discriminator
        set_model_requires_grad(netG, flag=False)
        set_model_requires_grad(netD, flag=True)

        temp_d_loss = []
        temp_d_train_prob_fake = []
        temp_d_train_prob_real = []
        temp_d_loss_reg = []

        data_itr = iter(data_loader)

        for ind in range(disc_train_num):
            real_wave, real_pga = next(data_itr)
            # test_wave = real_wave.detach().numpy()
            # test_wave = test_wave[2]
            # test_wave = test_wave.squeeze()
            # plt.plot(time, test_wave[2, :])
            # plt.show()
            # print(test_wave.shape)
            # exit()

            real_wave = real_wave.to(device)
            real_pga = real_pga.to(device)

            train_z = torch.randn(real_wave.shape[0], z_dim).to(device)
            fake_wave, fake_style, fake_pga = netG(train_z, is_train=True)

            real_d_out = netD(real_wave, real_pga)
            fake_d_out = netD(fake_wave.detach(), fake_pga.detach())

            d_loss = discriminator_loss(disc_fake_wave=fake_d_out, disc_real_wave=real_d_out)

            netD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            temp_d_loss.append(d_loss.item())
            temp_d_train_prob_fake.append(torch.sigmoid(fake_d_out).mean().item())
            temp_d_train_prob_real.append(torch.sigmoid(real_d_out).mean().item())

        d_loss_list.append(np.mean(temp_d_loss))
        d_train_prob_list_real.append(np.mean(temp_d_train_prob_real))
        d_train_prob_list_fake.append(np.mean(temp_d_train_prob_fake))

        # Normalization
        if epoch % disc_regularization_interval == 0:
            for _ in range(disc_train_num):
                real_wave, real_pga = next(data_itr)
                real_wave = real_wave.to(device)
                real_pga = real_pga.to(device)

                real_wave.requires_grad = True
                real_d_out = netD(real_wave, real_pga)
                d_reg = discriminator_loss_r1(disc_real_wave=real_d_out, reals_f=real_wave)

                netD.zero_grad()
                d_reg.backward()
                optimizerD.step()

                temp_d_loss_reg.append(d_reg.item())

            d_loss_reg_list.append(np.mean(temp_d_loss_reg))
        else:
            d_loss_reg_list.append(0)

        # Generator
        set_model_requires_grad(netG, flag=True)
        set_model_requires_grad(netD, flag=False)

        temp_g_loss = []
        temp_g_loss_reg = []
        temp_g_train_prob_fake = []

        for ind in range(gen_train_num):
            train_z = torch.randn(size=(batch_size, z_dim)).to(device)
            fake_wave, fake_style, fake_pga = netG(train_z, is_train=True)
            fake_d_out = netD(fake_wave, fake_pga)

            if epoch == 0 and ind == 0:
                print("gen wave size: {}".format(fake_wave.shape))
                print("gen label size: {}".format(fake_pga.shape))
                print("disc out size: {}".format(fake_d_out.shape))

            g_loss = generator_loss(fake_d_out)

            netG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            temp_g_loss.append(g_loss.item())
            temp_g_train_prob_fake.append(torch.sigmoid(fake_d_out).mean().item())

        g_loss_list.append(np.mean(temp_g_loss))
        g_train_prob_list.append(np.mean(temp_g_train_prob_fake))

        # Normalization
        if epoch % gen_regularization_interval == 0:
            for _ in range(gen_train_num):
                train_z = torch.randn(size=(batch_size, z_dim)).to(device)
                fake_wave, fake_style, _ = netG(train_z, is_train=True)
                gen_reg, _ = loss_func_g_reg(fake_wave_f=fake_wave, fake_style_f=fake_style)
                netG.zero_grad()
                gen_reg.backward()
                optimizerG.step()

                temp_g_loss_reg.append(gen_reg.item())

            g_loss_reg_list.append(np.mean(temp_g_loss_reg))
        else:
            g_loss_reg_list.append(0)

        if epoch % 100 == 0:
            netG.eval()

            fake_wave_valid, _, fake_pga_valid = netG(fixed_noise, is_train=False)
            out_data = fake_wave_valid.to("cpu").detach().numpy().squeeze()
            out_pga = fake_pga_valid.to("cpu").detach().numpy().squeeze()

            np.save(out_dir + "/wave_epoch_" + str(epoch) + ".npy", out_data)
            np.save(out_dir + "/log_pga_epoch_" + str(epoch) + ".npy", out_pga)

            netG.train()

        print("[Epoch: {}/{}] [D loss: {}] [G loss: {}]".format(epoch, num_epoch, g_loss_list[-1], d_loss_list[-1]))

        if (epoch + 1) % 5000 == 0:
            temp_state_dict_G = netG.state_dict()
            temp_out_path_G = out_dir + "model_G_epoch_" + str(epoch) + ".pth"
            torch.save(temp_state_dict_G, temp_out_path_G)

            temp_state_dict_D = netD.state_dict()
            temp_out_path_D = out_dir + "model_D_epoch_" + str(epoch) + ".pth"
            torch.save(temp_state_dict_D, temp_out_path_D)

    g_loss_list = np.array(g_loss_list)
    g_loss_reg_list = np.array(g_loss_reg_list)
    g_train_prob_list = np.array(g_train_prob_list)
    d_train_prob_list_real = np.array(d_train_prob_list_real)
    d_train_prob_list_fake = np.array(d_train_prob_list_fake)
    d_loss_list = np.array(d_loss_list)
    d_loss_reg_list = np.array(d_loss_reg_list)

    out_mat_1 = np.stack(
        [
            np.arange(num_epoch),
            g_loss_list,
            d_loss_list,
            g_loss_reg_list,
            d_loss_reg_list,
            g_train_prob_list,
            d_train_prob_list_real,
            d_train_prob_list_fake,
        ],
        axis=1,
    )
    col_names_1 = ["Epoch", "g_loss", "d_loss", "g_reg_loss", "d_reg_loss", "g_prob", "d_prob_real", "d_prob_fake"]
    df1 = pd.DataFrame(out_mat_1, columns=col_names_1)
    df1.to_csv(out_dir + "results_all.csv", index=False)
