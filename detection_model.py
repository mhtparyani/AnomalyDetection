import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv('health_data.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert to PyTorch tensors
X = torch.tensor(data_scaled, dtype=torch.float32)
dataset = TensorDataset(X, X)  # Autoencoder targets are the inputs themselves
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 50
for epoch in range(n_epochs):
    for inputs, _ in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'autoencoder_model.pth')

model.eval()
example_input = torch.rand(1, 4)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("autoencoder_model_mobile.pt")

# Load the TorchScript model
model = torch.jit.load('autoencoder_model_mobile.pt')

# Verify that the model works
example_input = torch.rand(1, 4)
output = model(example_input)
print(output)
