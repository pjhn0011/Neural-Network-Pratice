import torch
import torch.nn.functional as F

# Set device
# device can map the variable depending on the environment (CPU, GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
# can make same random variable
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

# Setup the input
X = torch.FloatTensor([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]).to(device)
Y = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]).to(device)

# Setup the model
layer = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=3, bias=True),
    torch.nn.ReLU(),
).to(device)

# Setup the optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

# Training
for step in range(100):
    # Extract output of single layer
    hypothesis = layer(X)
    # Calculate cost
    # 0: column-wise, 1: row-wise
    cost = (-Y.detach() * F.log_softmax(hypothesis, dim=1)).sum(dim=1).mean()
    # Softmax
    prob = hypothesis.softmax(dim=1)
    # Gradient initialization
    optimizer.zero_grad()
    # Calculate gradient
    cost.backward()
    # Update parameters
    optimizer.step()

    if step % 20 == 0:
        print(f'{step}: Loss: {cost.item()}')

# Evaluation
print('-------------------------')
for iter, test_input in enumerate(X):
    with torch.no_grad():
        hypothesis = layer(test_input)
        print(f'Iteration: {iter}')
        print(f'Input: {test_input.detach().cpu().numpy()}')
        print(f'output: {hypothesis.detach().cpu().numpy()}')
        print(f'ground-truth: {Y[iter].detach().cpu().numpy()}')
        print('-------------------------')

