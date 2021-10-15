import torch.nn as nn
import learn2learn as l2l

'''Convnet for miniImageNet'''


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim,
                                                  hidden=hid_dim,
                                                  channels=x_dim,
                                                  max_pool=True)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
