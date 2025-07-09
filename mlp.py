import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --------- CONFIGURATION ---------
input_dim = 20
hidden_dim = 64
num_hidden_layers = 3
output_dim = 5
use_dropout = True
dropout_prob = 0.5
batch_size = 64
num_epochs = 20
learning_rate = 0.001
test_split = 0.2
# ---------------------------------

# --------- DATA GENERATION ---------
X, y = make_classification(n_samples=5000, n_features=input_dim, n_classes=output_dim, n_informative=10)
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
test_size = int(len(dataset) * test_split)
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# -----------------------------------

# --------- MODEL DEFINITION ---------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim, use_dropout=False, dropout_prob=0.5):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, hidden_dim, num_hidden_layers, output_dim, use_dropout, dropout_prob)
# -----------------------------------

# --------- TRAINING SETUP ---------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
# ----------------------------------

# --------- TRAINING LOOP ---------
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
# ---------------------------------