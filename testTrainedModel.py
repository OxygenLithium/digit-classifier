import torch
from torch import nn, optim
from torchvision import datasets, transforms


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

model = torch.load('trainedModel', weights_only=False)
model.to(device)
model.eval()

rt = 0
dataiter = iter(loader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

for j in range(10000):
    imageTensor = images[j].view(1, 784).to(device)

    with torch.no_grad():
        logResult = model(imageTensor)

    result = torch.exp(logResult)
    _, predicted = torch.max(result, 1)

    if predicted.item() == labels[j].item():
        rt += 1

    if j % 100 == 0:
        print(j)

print("Correct Results (Out of 10000):")
print(rt)
