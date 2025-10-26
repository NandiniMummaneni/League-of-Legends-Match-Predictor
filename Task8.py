# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('league_data.csv')
except FileNotFoundError:
    print("league_data.csv not found. Creating sample data...")
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    data = {
        'kills': np.random.randint(0, 25, n_samples),
        'deaths': np.random.randint(0, 15, n_samples),
        'assists': np.random.randint(0, 20, n_samples),
        'gold': np.random.randint(5000, 30000, n_samples),
        'damage': np.random.randint(8000, 50000, n_samples)
    }
    # Create win based on performance
    win_prob = (data['kills'] - data['deaths'] + data['assists']/2) / 30 + 0.5
    win_prob = np.clip(win_prob, 0.1, 0.9)
    data['win'] = np.random.binomial(1, win_prob, n_samples)
    df = pd.DataFrame(data)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Split into features (X) and target (y)
# Assuming the target column is named 'win' (1 = win, 0 = loss)
X = df.drop(columns=['win'])
y = df['win']

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Verify shapes
print("\nShapes of tensors:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_test:", X_test_tensor.shape)
print("y_test:", y_test_tensor.shape)