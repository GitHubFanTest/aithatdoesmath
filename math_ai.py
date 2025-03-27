import torch
import torch.nn as nn

# Step 1: Generate the dataset for addition
def generate_addition_data(num_samples=1000):
    inputs = torch.randint(0, 100, (num_samples, 2), dtype=torch.float32)  # Random pairs of numbers
    targets = inputs[:, 0] + inputs[:, 1]  # Sum of the numbers
    return inputs, targets

# Step 2: Define the Model
class MathModel(nn.Module):
    def __init__(self):
        super(MathModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input: 2 numbers, Output: 64 neurons
        self.fc2 = nn.Linear(64, 1)  # Final output: 1 number (result of addition)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Create a function to train the model
def train_model(model, inputs, targets, num_epochs=1000, lr=0.001):
    criterion = nn.MSELoss()  # For regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)  # Remove extra dimension in outputs
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 4: Test the trained model
def test_model(model):
    model.eval()
    test_input = torch.tensor([[30.0, 40.0]])  # Test input
    with torch.no_grad():
        prediction = model(test_input)
    rounded_prediction = round(prediction.item())  # Round to nearest integer
    print(f"Prediction: {rounded_prediction}, Expected: 70")

# Main script
if __name__ == "__main__":
    # Step 5: Generate training data
    inputs, targets = generate_addition_data()

    # Step 6: Instantiate and train the model
    model = MathModel()
    train_model(model, inputs, targets)

    # Step 7: Evaluate the model's performance
    test_model(model)
