import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MetalDataset(Dataset):
    """Dataset для нейронной сети"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.LongTensor(y.values if isinstance(y, pd.Series) else y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNeuralNetwork(nn.Module):
    """Простая нейронная сеть для бинарной/многоклассовой классификации"""
    def __init__(self, input_size, num_classes=2, dropout_rate=0.3, hidden_sizes=[128, 64, 32]):
        super(SimpleNeuralNetwork, self).__init__()
        
        # Гибкая архитектура с настраиваемыми размерами слоев
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, num_classes)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return x


