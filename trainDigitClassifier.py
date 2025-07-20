import torch
from torch import nn, optim
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
mnistSet = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=transform)
loader = torch.utils.data.DataLoader(mnistSet, batch_size=128, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(64, 10))
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)
for i in range(16):
    runningLoss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        
        runningLoss += loss.item()
    else:
        print("Training loss: ", runningLoss/len(loader))

print("Reduced learning rate to 0.01")

optimizer = optim.SGD(model.parameters(), lr=0.01)
for i in range(8):
    runningLoss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        
        runningLoss += loss.item()
    else:
        print("Training loss: ", runningLoss/len(loader))

print("Reduced learning rate to 0.0025")

optimizer = optim.SGD(model.parameters(), lr=0.0025)
for i in range(8):
    runningLoss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
    
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
