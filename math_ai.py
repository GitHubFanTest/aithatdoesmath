import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Generate a dataset of 10,000 random (a, b) pairs and their sums
num_samples = 10000
x_train = []
y_train = []

for _ in range(num_samples):
    a = random.randint(0, 500)  # Random number between 0 and 500
    b = random.randint(0, 500)
    x_train.append([a, b])
    y_train.append([a + b])

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader for mini-batch training
batch_size = 32
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
class MathModel(nn.Module):
    def __init__(self):
        super(MathModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input: (a, b)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: sum(a, b)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = MathModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    for data, target in dataloader:
        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Print progress
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model
model.eval()
test_input = torch.tensor([[346.0, 464.0]], dtype=torch.float32)  # Example input
predicted = model(test_input).item()
print(f"Prediction: {predicted:.4f}, Expected: 810")
