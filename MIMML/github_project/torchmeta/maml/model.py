import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size),
        )

        # self.classifier = MetaLinear(hidden_size * (1 ** 2), out_features)
        self.classifier = MetaLinear(hidden_size * (5 ** 2), out_features)

    def forward(self, inputs, params=None):
        # print('inputs.size()', inputs.size())
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        # print('features.size()', features.size())
        features = features.view((features.size(0), -1))
        # print('features.size()', features.size())
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


if __name__ == '__main__':
    model = ConvolutionalNeuralNetwork(3, 5, hidden_size=64)
    print(model)

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
