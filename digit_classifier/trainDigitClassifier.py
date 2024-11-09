import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

mnistSet = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=transform)
loader = torch.utils.data.DataLoader(mnistSet, batch_size=128, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.Sigmoid(),
                      nn.Dropout(0.1),
                      nn.Linear(128, 64),
                      nn.Sigmoid(),
                      nn.Dropout(0.1),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.5)
for i in range(12):
    runningLoss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1)
    
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        
        runningLoss += loss.item()
    else:
        print("Training loss: ", runningLoss/len(loader))


optimizer = optim.SGD(model.parameters(), lr=0.1)
for i in range(6):
    runningLoss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1)
    
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        
        runningLoss += loss.item()
    else:
        print("Training loss: ", runningLoss/len(loader))

torch.save(model, 'trainedModel')
print("Successfully trained and saved")
