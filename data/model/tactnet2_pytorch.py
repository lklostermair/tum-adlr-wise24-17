# import torch.nn as nn
# import torch.nn.functional as F
 
# class tactnet2Siamese(nn.Module):
#     def __init__(self):
#         super(tactnet2Siamese, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, 32, (15,5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(10,1),
#             nn.Conv2d(32, 64, (15,5)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(10,1),
#             nn.Conv2d(64, 128, (15,5)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(10,1),
#             nn.Dropout(p=0.8),
#             nn.Linear(128, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 36),
#         )
    
