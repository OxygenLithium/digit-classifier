import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
testset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=True)

model = torch.load('trainedModel', weights_only=False)

dataiter = iter(loader)
images, labels = next(dataiter)

imageTensor = images[0].view(1, 784)

with torch.no_grad():
    logResult = model(imageTensor)

result = torch.exp(logResult)

print(labels[0].item())

resultList = result.tolist()
for i in range(10):
    print(i, ": ", round(resultList[0][i],3))
