import torch
import torch.nn as nn
import torch.optim as optim

# Define Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# Initialize model, loss, optimizer
input_dim = 5  # Replace with number of features in your dataset
model = LogisticRegressionModel(input_dim=input_dim)

# Binary Cross Entropy loss
criterion = nn.BCELoss()

# SGD optimizer for training the model
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Logistic Regression model initialized successfully!")
print(model)