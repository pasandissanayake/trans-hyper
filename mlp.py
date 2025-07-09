import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import os

from utils import Config, ConfigObject
from datahandles import TabLLMDataObject, CombinedTabLLMTextDataset

# --------- CONFIGURATION ---------
input_dim = 8
hidden_dim = 10
num_hidden_layers = 2
output_dim = 2
use_dropout = False
dropout_prob = 0.5
batch_size = 64
num_epochs = 20
learning_rate = 0.001
cfg_path = 'cfgs/bert.yaml'
# ---------------------------------



# --------- PARSING ARGS AND CONFIGS --------
def make_cfg(path):
    cfg = Config(path)
    setattr(cfg, "env", ConfigObject())
    setattr(cfg.env, "total_gpus", ConfigObject(torch.cuda.device_count())) # type: ignore
    return cfg
# -------------------------------------------




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


# --------- DATA GENERATION ---------
cfg = make_cfg(cfg_path)
tabllm_do = TabLLMDataObject(cfg)
train_dataset = CombinedTabLLMTextDataset(cfg, 'train', tabllm_do.split_datapoints)
test_dataset = CombinedTabLLMTextDataset(cfg, 'test', tabllm_do.split_datapoints)

test_size = len(test_dataset)
train_size = len(train_dataset)
print(f"test size:{test_size}, train size:{train_size}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# -----------------------------------

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
        for batch in loader:
            inputs = batch['x']
            labels = batch['y']
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
# ----------------------------------

# --------- TRAINING LOOP ---------
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # print(f"batch:{batch}")
        inputs = batch['x']
        labels = batch['y']
        # print(f"inputs:{inputs}, labels:{labels}")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
# ---------------------------------