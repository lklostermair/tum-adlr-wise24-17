import torch
import torch.nn as nn 
import torch.nn.functional as F 

class TactNetII(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length):
        super(TactNetII, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(15, 5))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(15, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(15, 5))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.dropout3 = nn.Dropout(0.1)

        self.dropout = nn.Dropout(0.8)

        # Use the user-provided sequence_length to derive flattened size
        self.flattened_size = self._get_flatten_size(sequence_length, input_channels)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flatten_size(self, seq_len, input_channels):
        with torch.no_grad():
            # Create a dummy tensor of shape [batch=1, channels=input_channels, seq_len, 16]
            x = torch.zeros(1, input_channels, seq_len, 16)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            return x.numel()  # or x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x
