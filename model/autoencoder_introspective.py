from model import *
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import os

'''
This Autoencoder model is adapted from marian42's autoencoder "https://github.com/marian42/shapegan", following the
idea of IntroVAE "https://github.com/hhb072/IntroVAE"
'''


model_path = "model"
chckpt = os.path.join(model_path, 'chckpt')
lantent_code_size = 256
model_complexity = 48
standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function
    def forward(self, x):
        return self.function(x)

class SavableModule(nn.Module):
    def __init__(self, filename):
        super(SavableModule, self).__init__()
        self.filename = filename

    def get_filename(self, epoch=None, filename=None):
        if filename is None:
            filename = self.filename
        if epoch is None:
            return os.path.join(model_path, filename)
        else:
            filename = filename.split('.')
            filename[-2] += '-epoch-{:05d}'.format(epoch)
            filename = '.'.join(filename)
            return os.path.join(chckpt, filename)

    def load(self, epoch=None):
        print(self.get_filename(epoch=epoch))
        self.load_state_dict(torch.load(self.get_filename(epoch=epoch)), strict=False)

    def save(self, epoch=None):
        if epoch is not None and not os.path.exists(chckpt):
            os.mkdir(chckpt)
        torch.save(self.state_dict(), self.get_filename(epoch=epoch))

    @property
    def device(self):
        return next(self.parameters()).device


class Autoencoder(SavableModule):
    def __init__(self, vox_res = 32, is_variational = True):
        super(Autoencoder, self).__init__(filename="IntroVAE_final_param_{}vx-{:d}.to".format(vox_res,lantent_code_size))
        self.voxel_size = vox_res
        self.is_variational = is_variational

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 1 * model_complexity, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(1 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels = 1 * model_complexity, out_channels = 2 * model_complexity, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(2 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels = 2 * model_complexity, out_channels = 4 * model_complexity, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(4 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels = 4 * model_complexity, out_channels = lantent_code_size * 2, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(lantent_code_size * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            nn.Linear(in_features = lantent_code_size * 2, out_features=lantent_code_size)
        )

        if is_variational:
            self.encoder.add_module('vae-bn', nn.BatchNorm1d(lantent_code_size))
            self.encoder.add_module('vae-lr', nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.encode_mean = nn.Linear(in_features=lantent_code_size, out_features=lantent_code_size)
            self.encode_log_variance = nn.Linear(in_features=lantent_code_size, out_features=lantent_code_size)
        self.decoder = nn.Sequential(
            nn.Linear(in_features = lantent_code_size, out_features=lantent_code_size * 2),
            nn.BatchNorm1d(lantent_code_size * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Lambda(lambda x: x.reshape(-1, lantent_code_size * 2, 1, 1, 1)),
            nn.ConvTranspose3d(in_channels = lantent_code_size * 2, out_channels = 4 * model_complexity, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(4 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(in_channels = 4 * model_complexity, out_channels = 2 * model_complexity, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(2 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(in_channels = 2 * model_complexity, out_channels = 1 * model_complexity, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(1 * model_complexity),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(in_channels = 1 * model_complexity, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        )
        self.cuda()

    def encode(self, x, return_mean_and_log_variance = False):
        x = x.reshape((-1, 1, 32, 32, 32))
        x = self.encoder(x)

        if not self.is_variational:
            return x

        mean = self.encode_mean(x).squeeze()

        if self.training or return_mean_and_log_variance:
            log_variance = self.encode_log_variance(x).squeeze()
            standard_deviation = torch.exp(log_variance * 0.5)
            eps = standard_normal_distribution.sample(mean.shape).to(x.device)

        if self.training:
            x = mean + standard_deviation * eps
        else:
            x = mean

        if return_mean_and_log_variance:
            return x, mean, log_variance
        else:
            return x

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim = 0)
        x = self.decoder(x)
        return x.squeeze()

    def forward(self, x):
        if not self.is_variational:
            z = self.encode(x)
            x = self.decode(z)
            return x

        z, mean, log_variance = self.encode(x, return_mean_and_log_variance = True)
        x = self.decode(z)
        return mean, log_variance, z, x
