import torch.nn as nn
import torch

class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass, bias=True)

    def forward(self, input, use_low=None):
        feature = self.feature(input, use_low)
        out = self.fc(feature[-1])
        return out

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs, use_low=None):
        return self.feature(inputs, use_low)

