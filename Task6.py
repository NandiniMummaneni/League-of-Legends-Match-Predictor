import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv('league_data.csv')
X = df.drop('win', axis=1).values   
y = df['win'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Train model
model = LogisticRegressionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

# Save model state dictionary
torch.save(model.state_dict(), "logistic_model.pth")
print("Model saved successfully!")

# Load model into a new instance
loaded_model = LogisticRegressionModel(X_train.shape[1])
loaded_model.load_state_dict(torch.load("logistic_model.pth"))
loaded_model.eval()
print("Model loaded successfully and set to evaluation mode.")

# Evaluate loaded model
with torch.no_grad():
    y_pred_loaded = loaded_model(X_test).round()
    acc = accuracy_score(y_test, y_pred_loaded)

print(f"Accuracy of loaded model: {acc * 100:.2f}%")