import torch
import torch.nn as nn
import torch.optim as optim
import loader
import torch.nn.functional as F
import torch.optim as optim
import model

"""
model = nn.Sequential(
    # nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(3, 6, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 10, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(10, 16, kernel_size=10),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # nn.Linear(16 * 58 * 58, 512),
    # nn.ReLU(),
    # nn.Linear(512, 10),
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        # return
        x = x.view(-1, 16 * 62 * 62)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""

net = model.Net().cuda()


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

n_epochs = 20

one_epoch_loss = 10.0  # set a big value so that the loop can work in the beginning
dataloader = loader.dataloader
epoch = 0

while one_epoch_loss > 1.3:
    # for epoch in range(n_epochs):
    running_loss = 0.0
    one_epoch_loss = 0.0
    for i, sample_batched in enumerate(dataloader):
        # print(i)
        image = sample_batched["image"]
        kind = sample_batched["kind"]
        # image, kind = sample_batched
        # print("sample_batched")
        # print(sample_batched)
        optimizer.zero_grad()

        out = net(image.cuda())
        loss = loss_fn(out, kind.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        one_epoch_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    one_epoch_loss = one_epoch_loss / 2400
    print("epoch %d, average loss %.3f" % (epoch + 1, one_epoch_loss))
    # one_epoch_loss = 0.0
    epoch += 1
# sample_batched["image"].size()


torch.save(net.state_dict(), model.PATH)
