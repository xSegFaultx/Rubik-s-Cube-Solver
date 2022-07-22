import torch
from torch.utils.data import Dataset


class CubeDataset(Dataset):
    def __init__(self, data, label):
        self.x = data
        self.y = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        sample = {"x": x, "y": y}
        return sample
