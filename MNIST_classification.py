import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# Reproducibility
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# image set
train_X = torchvision.datasets.MNIST('./data', True, transform=trans, download=False)
test_X = torchvision.datasets.MNIST('./data', False, transform=trans, download=False)

# data loader
train_loader = DataLoader(train_X, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_X, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

# model
layer = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=784, out_features=256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=10, bias=True),
).to(device)
print(layer)

# optimizer
optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

# training
for epoch in range(15):
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.float().to(device), labels.long().to(device)

        # Extract output of single layer
        hypothesis = layer(images)

        # Calculate cross-entropy loss
        # cost = (-labels.detach() * F.log_softmax(hypothesis, dim=1)).sum(dim=1).mean()
        cost = F.cross_entropy(input=hypothesis, target=labels)

        # gradient initialization
        optimizer.zero_grad()

        # Calculate gradient
        cost.backward()

        # Update parameters
        optimizer.step()

        # Calculate accuracy
        prob = hypothesis.softmax(dim=1)
        pred = prob.argmax(dim=1)
        acc = pred.eq(labels).float().mean()
        if (idx+1) % 128 == 0:
            print(f'TRAIN-Iteration: {idx+1}, Loss: {cost.item()}, Accuracy: {acc.item()}')

    # evaluation
    with torch.no_grad():
        acc = 0
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.float().to(device), labels.long().to(device)

            # Extract output of single layer
            hypothesis = layer(images)

            # Calculate cross-entropy loss
            cost = F.cross_entropy(input=hypothesis, target=labels)

            # Calculate accuracy
            prob = hypothesis.softmax(dim=1)
            pred = prob.argmax(dim=1)
            acc += pred.eq(labels).float().mean()
        print(f'TEST-Accuracy: {acc/len(test_loader)}')