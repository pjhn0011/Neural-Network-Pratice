import torch

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

# input
X = torch.FloatTensor([[1], [2], [3]]).to(device)
Y = torch.FloatTensor([[1], [2], [3]]).to(device)

# model
layer = torch.nn.Linear(in_features=1, out_features=1, bias=True).to(device)

# Setup criterion
criterion = torch.nn.MSELoss()

# Setup optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

# training
for step in range(100):
    # Extract output of single layer
    hypothesis = layer(X)
    # Calculate cost
    cost = criterion(hypothesis, Y)
    # gradient initialization
    optimizer.zero_grad()
    # Calculate gradient
    cost.backward()
    # Update parameters
    optimizer.step()

    if step % 20 == 0:
        print(f'{step}: Loss: {cost.item()}')

print('-------------------------')
# print parameters
for name, param in layer.named_parameters():
    print(f'{name}: {param.detach().cpu().numpy()}')

# Accuracy computation
X_t = torch.FloatTensor([[5], [2.5]]).to(device)

print('-------------------------')
for iter, test_input in enumerate(X_t):
    with torch.no_grad():
        hypothesis = layer(test_input)
        print(
            f'Iteration: {iter},'
            f'Input: {test_input.detach().cpu().numpy()},'
            f'output: {hypothesis.detach().cpu().numpy()}'
        )

