import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

# Setup the transformation
trans = transforms.Compose([
    transforms.ToTensor(),  # PIL to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalize the image with mean, std value
])

# Load the image set
# Download the train and test dataset, respectively.
# train image set (60,000, 28, 28), test image set (10,000, 28, 28)
train_X = torchvision.datasets.MNIST('./data', True, transform=trans, download=False)
test_X = torchvision.datasets.MNIST('./data', False, transform=trans, download=False)

# Setup the loader
# for train data, data random shuffle and drop_last are used.
# for test data, data random shuffle and drop_last are not used for same comparison
# pin_memory option can enhance the GPU utility
train_loader = DataLoader(train_X, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_X, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

# Setup the model
# input data transform to one-dimensional vector
layer = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=784, out_features=256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=10, bias=True),  # the output dimension is the number of class
).to(device)
print(layer)  # print the defined model

# Setup the optimizer
optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

# Training
# 15 epoch: each epoch iterate N / batch size (N: the number of data)
for epoch in range(15):
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.float().to(device), labels.long().to(device)

        # Extract output of single layer
        # output dimension will be (M, K)
        # M: batch size, K: the number of class
        hypothesis = layer(images)

        # Calculate cross-entropy loss
        # input shape: [M,K], target shape: [M,]
        # label data is scalar value of class (0~9)
        # detach() can prevent the backpropagation for label data
        cost = F.cross_entropy(
            input=hypothesis,
            target=labels.detach())

        # Gradient initialization
        optimizer.zero_grad()
        # Calculate gradient
        cost.backward()
        # Update parameters
        optimizer.step()

        # Calculate accuracy
        # 0: column-wise, 1: row-wise
        prob = hypothesis.softmax(dim=1)
        # Extract index of maximum possibility component
        pred = prob.argmax(dim=1)
        # Compare the equality between prediction and ground-truth
        acc = pred.eq(labels).float().mean()
        # display at every 128 iteration
        if (idx+1) % 128 == 0:
            print(f'TRAIN-Iteration: {idx+1}, Loss: {cost.item()}, Accuracy: {acc.item()}')

    # Evaluation
    # The parameters must be fixed during the inference
    with torch.no_grad():
        acc = 0
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.float().to(device), labels.long().to(device)

            # Extract output of model
            hypothesis = layer(images)

            # Calculate cross-entropy loss
            cost = F.cross_entropy(
                input=hypothesis,
                target=labels)

            # Calculate accuracy
            # 0: column-wise, 1: row-wise
            prob = hypothesis.softmax(dim=1)
            # Extract index of maximum possibility component
            pred = prob.argmax(dim=1)
            # Compare the equality between prediction and ground-truth
            acc += pred.eq(labels).float().mean()
        print(f'TEST-Accuracy: {acc/len(test_loader)}')
