import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loader
import model

net = model.Net().cuda()
net.load_state_dict(torch.load(model.PATH))


correct = 0
total = 0
with torch.no_grad():
    for data in loader.validloader:
        # print("data")
        # print(data)
        images = data["image"].cuda()
        kinds = data["kind"].cuda()
        outputs = net(images)
        # print("kinds")
        # print(kinds)
        # print("outputs")
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print("predicted")
        # print(predicted)
        total += kinds.size(0)
        correct += (predicted == kinds).sum().item()

print("Accuracy of the network on the valid images: %d %%" % (100 * correct / total))

