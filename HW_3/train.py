import torch
import torch.nn as nn
import torch.optim as optim
import loader

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
)

learning_rate = 1e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 100

dataloader = loader.dataloader
for epoch in range(n_epochs):
    for index, sample_batched in enumerate(dataloader):
        image = sample_batched["image"]
        kind = sample_batched["kind"][0]
        print(image)
        out = model(image)
        loss = loss_fn(out, kind)
        break
# sample_batched["image"].size()

