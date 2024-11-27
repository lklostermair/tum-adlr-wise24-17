# Model Definition
class MCTestTacNet(nn.Module):
    def __init__(self):
        super(MCTestTacNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(15, 5))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(15, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(15, 5))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(10, 1), stride=1)
        self.dropout = nn.Dropout(0.8)

        # Dummy forward pass to calculate the flattened size
        self.flattened_size = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 36)

    def _get_flatten_size(self): # helper function for dynamic sizing of linear layer
        with torch.no_grad():
            x = torch.zeros(1, 1, 1000, 16)  # example
            x = self.pool3(self.bn3(self.conv3(self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(x)))))))))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn4(self.fc1(self.dropout(x))))
        x = self.fc2(x)  # no softmax bc cross entropy loss includes it
        return x
