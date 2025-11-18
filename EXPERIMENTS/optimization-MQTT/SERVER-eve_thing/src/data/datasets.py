import torch
from torch.utils.data import Dataset

class IMUWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N,T,C)
        self.y = torch.tensor(y, dtype=torch.long)     # encoded labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]