# Neural-Network-Pratice

This is for basic pytorch code for beginner.

requirements
- pytorch 1.8.0
- torchvision 0.9.0

If you already downloaded the dataset, change the flag for download=False. If not download=True.

train_X = torchvision.datasets.MNIST('./data', True, transform=trans, download=False)
test_X = torchvision.datasets.MNIST('./data', False, transform=trans, download=False)
