import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
testset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=True)

model = torch.load('trainedModel', weights_only=False)

rt = 0
for j in range(1000):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    imageTensor = images[0].view(1, 784)

    with torch.no_grad():
        logResult = model(imageTensor)

    result = torch.exp(logResult)

    resultList = result.tolist()

    currMax = 0
    currIndex = 0
    for i in range(10):
        if resultList[0][i] > currMax:
            currMax = resultList[0][i]
            currIndex = i
    if currIndex == labels[0].item():
        rt += 1

print("Correct Results (Out of 1000):")
print(rt)
