import torch
from torch import nn
import torch.nn.functional as F


class VGGlikeNet(nn.Module):
    def __init__(
            self,
            num_filters = [4, 8, 16]
    ):
        super(VGGlikeNet, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= num_filters[0], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[0], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[1], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[2], kernel_size=(3,3), padding=(1,1), bias=True),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.LazyLinear(out_features=10)
        self.output = nn.Linear(in_features=10, out_features=2)

    def forward(self, input):
        x = self.conv_block1(input)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
        
        x = self.conv_block2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
        
        x = self.conv_block3(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))

        x = self.flatten(x)
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        
        x = self.output(x)
        
        return x
        
class SpectrogramCNN(nn.Module):
    def __init__(
            self,
            num_filters = [6, 16, 120]
    ):
        super(SpectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=(5,5), padding=(2,2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=(5,5), padding=(2,2), bias=True)
        self.conv3 = nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=(5,5), padding=(2,2), bias=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=84)
        self.output = nn.Linear(in_features=84, out_features=2)

    def forward(self, input: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(input)), kernel_size=(2,2), stride=(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2,2), stride=(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2,2), stride=(2,2))

        x = self.flatten(x)
        x = F.dropout(x, 0.5)
        x = F.relu(F.relu(self.fc1(x)))
        x = F.dropout(x, 0.5)
        x = self.output(x)

        return x
