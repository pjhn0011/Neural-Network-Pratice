import torch

# Set device
# device can map the variable depending on the environment (CPU, GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
# can make same random variable
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

# Setup the input
X = torch.FloatTensor([[1], [2], [3]]).to(device)
Y = torch.FloatTensor([[1], [2], [3]]).to(device)

# Setup the model
layer = torch.nn.Linear(
    in_features=1,
    out_features=1,
    bias=True).to(device)

# Setup criterion
# mean squared error = mean((X-Y)^2)
criterion = torch.nn.MSELoss()

# Setup optimizer
optimizer = torch.optim.SGD(
    params=layer.parameters(),  # parameters to update
    lr=0.1)  # initial learning rate

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

