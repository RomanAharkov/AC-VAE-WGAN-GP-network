import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init


class Generator(nn.Module):
    def __init__(self, class_num: int, latent_dim: int, c: int, mode="PCA"):
        super().__init__()
        self.class_num = class_num
        n = math.floor(c/16)  # 12
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.input_bn = nn.BatchNorm1d(32 * n * 16)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        if mode == "VAE":
            self.input_layer = nn.Linear(latent_dim+class_num, 32*n*16, bias=False)
        elif mode == "PCA":
            self.input_layer = nn.Linear(130+class_num, 32*n*16, bias=False)
        self.conv1 = nn.ConvTranspose1d(512, 256, 3, 2, bias=False)  # 12 x 512 -> 25 x 256
        if c == 200:
            self.conv2 = nn.ConvTranspose1d(256, 128, 3, 2, padding=1, output_padding=1, bias=False)  # 25 x 256 -> 50 x 128
        elif c == 204:
            self.conv2 = nn.ConvTranspose1d(256, 128, 3, 2, bias=False)  # 25 x 256 -> 51 x 128
        self.conv3 = nn.ConvTranspose1d(128, 64, 3, 2, padding=1, output_padding=1, bias=False)  # 50 (51) x 128 -> 100 (102) x 64
        self.conv4 = nn.ConvTranspose1d(64, 1, 3, 2, padding=1, output_padding=1, bias=False)  # 100 (102) x 64 -> 200 (204) x 1

    def forward(self, noise, c, pca=None):
        """
        noise = (batch_size, latent_dim)
        c = (batch_size, 1)
        pca = (batch_size, 30) | None
        """
        c = c.squeeze(1).long()  # (batch_size)
        c = F.one_hot(c, num_classes=self.class_num)

        if pca is not None:
            x = torch.cat([noise, pca, c], dim=1)
        else:
            x = torch.cat([noise, c], dim=1)
        x = self.relu(self.input_bn(self.input_layer(x)))  # (batch_size, 32*16*n)
        x = x.view(x.size(0), 512, -1)
        x = self.relu(self.bn1(self.conv1(x)))  # (batch_size, 256, 25)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, 50/51)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch_size, 64, 100/102)
        x = self.tanh(self.conv4(x))  # (batch_size, 1, 200/204)
        return x


class Discriminator(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(256)
        self.bn4 = nn.InstanceNorm1d(512)
        self.conv1 = nn.Conv1d(1, 64, 3, 2, padding=1, bias=False)  # 200 (204) x 1 -> # 100 (102) x 64
        self.conv2 = nn.Conv1d(64, 128, 3, 2, padding=1, bias=False)  # 100 (102) x 64 -> 50 (51) x 128
        if c == 200:
            self.conv3 = nn.Conv1d(128, 256, 3, 2, padding=1, bias=False)  # 50 x 128 -> 25 x 256
        elif c == 204:
            self.conv3 = nn.Conv1d(128, 256, 3, 2, bias=False)  # 51 x 128 -> 25 x 256
        self.conv4 = nn.Conv1d(256, 512, 3, 2, bias=False)  # 25 x 256 -> 12 x 512
        self.fc = nn.Linear(12*512, 1, bias=False)

    def forward(self, x):
        """
        x = (batch_size, 1, c)
        """
        x = self.leaky_relu(self.conv1(x))  # (batch_size, 64, c/2)
        x = self.leaky_relu(self.bn2(self.conv2(x)))  # (batch_size, 128, c/4)
        x = self.leaky_relu(self.bn3(self.conv3(x)))  # (batch_size, 256, c/8)
        x = self.leaky_relu(self.bn4(self.conv4(x)))  # (batch_size, 512, c/16)
        x = self.flatten(x)  # (batch_size, 32c)
        x = self.fc(x)  # (batch_size, 1)
        return x


class Classifier(nn.Module):
    def __init__(self, class_num: int):
        super().__init__()
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(1, 64, 15, 15, padding=7)  # 200 (204) x 1 -> 14 x 64
        self.fc = nn.Linear(64 * 14, class_num)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x = (batch_size, 1, c)
        """
        x = self.tanh(self.conv1(x))  # (batch_size, 64, c/15)
        x = self.flatten(x)  # (batch_size, 64/15*c)
        x = self.fc(x)  # (batch_size, num_classes)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, c: int):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(1, 32, 3, 2, padding=1, bias=False)  # 200 (204) x 1 -> # 100 (102) x 32
        self.conv2 = nn.Conv1d(32, 64, 3, 2, padding=1, bias=False)  # 100 (102) x 32 -> 50 (51) x 64
        if c == 200:
            self.conv3 = nn.Conv1d(64, 128, 3, 2, padding=1, bias=False)  # 50 x 64 -> 25 x 128
        elif c == 204:
            self.conv3 = nn.Conv1d(64, 128, 3, 2, bias=False)  # 51 x 64 -> 25 x 128
        self.fc1 = nn.Linear(25*128, latent_dim, bias=False)
        self.fc2 = nn.Linear(25*128, latent_dim, bias=False)

    def forward(self, x):
        """
        x = (batch_size, 1, c)
        """
        x = self.leaky_relu(self.bn1(self.conv1(x)))  # (batch_size, 32, c/2)
        x = self.leaky_relu(self.bn2(self.conv2(x)))  # (batch_size, 64, c/4)
        x = self.leaky_relu(self.bn3(self.conv3(x)))  # (batch_size, 128, c/8)
        x = self.flatten(x)  # (batch_size, 16c)
        mu = self.fc1(x)  # (batch_size, latent_dim)
        log_s = self.fc2(x)  # (batch_size, latent_dim)
        return mu, log_s


class CNN1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.n1 = channels
        self.n5 = 16
        if channels == 200:
            self.k1 = 22
            self.n3 = 35
        else:
            self.k1 = 23
            self.n3 = 36
        self.k2 = 5
        self.n4 = 100

        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(1, 20, self.k1, 1, 0)
        self.pool = nn.MaxPool1d(kernel_size=self.k2, stride=self.k2)
        self.fc1 = nn.Linear(20 * self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def forward(self, x):
        """
        x = (batch_size, 1, c)
        """
        x = self.conv1(x)
        x = self.tanh(self.pool(x))
        x = x.view(x.shape[0], 20 * self.n3)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN3D(nn.Module):
    def __init__(self, num_classes: int, channels: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=2,
            kernel_size=(7, 3, 3),
            stride=1,
            padding=0
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=2,
            out_channels=8,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0
        )
        if channels == 200:
            self.fc1 = nn.Linear(8 * 1 * 1 * 192, 128)
        else:
            self.fc1 = nn.Linear(8 * 1 * 1 * 196, 128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (batch_size, 1, 200 (204), 5, 5)
        """
        x = self.conv1(x)  # (batch, 2, 3, 3, 194 (198))
        x = self.relu(x)

        x = self.conv2(x)  # (batch, 8, 1, 1, 192 (196))
        x = self.relu(x)

        x = x.view(x.shape[0], -1)  # (batch, 8*1*1*192 (196))

        x = self.fc1(x)  # (batch, 128)
        x = self.relu(x)

        x = self.fc2(x)  # (batch, num_classes)
        return x
