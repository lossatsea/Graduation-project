import torch
import torch.nn as nn
from utils import weight_initialize
import torchvision.transforms as transforms

class ResidualBlock_G(nn.Module):
    def __init__(self):
        super(ResidualBlock_G, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8)
        )

    def forward(self, x):
        out = self.block(x)
        return torch.add(x, out)

class UpscaleBlock(nn.Module):
    def __init__(self) :
        super(UpscaleBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.block(x)
        return out

class generator(nn.Module):
    def __init__(self, latent_dim, classes_dim, is_cat=True):
        super(generator, self).__init__()

        self.latent_dim = latent_dim

        self.block_input_1 = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
        )
        self.block_input_2 = nn.Sequential(
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )

        self.residual = self.make_layer(ResidualBlock_G, 16)

        self.block_mid = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )

        self.upsample = self.make_layer(UpscaleBlock, 3)

        self.block_output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

        weight_initialize(self)

    def make_layer(self, block, num):
        layers = []
        for _ in range(num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #first
        out = self.block_input_1(x)
        out = out.reshape((out.shape[0], 64, 16, 16))
        out = self.block_input_2(out)
        residual = out

        #residual
        out = self.residual(out)

        #after residual
        out = self.block_mid(out)
        out = torch.add(residual, out)
        
        #upsample
        out = self.upsample(out)
        
        #final
        out = self.block_output(out)
        
        return out

class ResiduaBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, last_kernel):
        super(ResiduaBlock_D, self).__init__()

        self.leakyReLu = nn.LeakyReLU(inplace=True)

        self.block1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                self.leakyReLu,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*2, kernel_size=last_kernel, stride=2, padding=1),
            self.leakyReLu
        )

    def forward(self, x):
        out = torch.add(x, self.block1(x))
        out = self.leakyReLu(out)

        out = torch.add(out, self.block1(out))
        out = self.leakyReLu(out)

        out = self.block2(out)
        return out

class discriminator(nn.Module):
    def __init__(self, in_channels, label_num):
        super(discriminator, self).__init__()

        self.block_input = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(inplace=True)
        )

        self.block_res = nn.Sequential(
            ResiduaBlock_D(32, 32, 4),
            ResiduaBlock_D(64, 64, 4),
            ResiduaBlock_D(128, 128, 3),
            ResiduaBlock_D(256, 256, 3),
            ResiduaBlock_D(512, 512, 3)
        )

        self.sigmoid = nn.Sigmoid()
        self.dense_judge = nn.Linear(1024 * 2 * 2, 1)
        self.dense_label = nn.Linear(1024 * 2 * 2, label_num)

        weight_initialize(self)

    def forward(self, img):
        out = self.block_input(img)
        out = self.block_res(out)

        out = out.view(out.shape[0], -1)

        out1 = self.dense_judge(out)
        judge = self.sigmoid(out1)

        out2 = self.dense_label(out)
        label = self.sigmoid(out2)

        return judge, label
