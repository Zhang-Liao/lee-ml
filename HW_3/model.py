import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = "./homework3_2.pth"

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=(1, 1))
        # self.conv3 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        # return
        x = x.view(-1, 16 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(1, 1))
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, padding=(1, 1), stride=(4, 4))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=(1, 1))

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=(1, 1))

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=(1, 1))

        # self.conv3 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 11)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        # return
        x = x.view(-1, 256 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(1, 1))
