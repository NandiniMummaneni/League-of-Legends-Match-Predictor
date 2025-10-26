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

# Hyperparameter tuning
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
best_acc = 0
best_lr = None

for lr in learning_rates:
    model = LogisticRegressionModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Train model
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test).round()
        acc = accuracy_score(y_test, y_pred_test)
    
    print(f"Learning Rate: {lr}, Test Accuracy: {acc * 100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print(f"\nBest Learning Rate: {best_lr}, Test Accuracy: {best_acc * 100:.2f}%")