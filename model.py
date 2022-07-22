import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20 * 24, 4096),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, 12)
        )

    def forward(self, x):
        x = x.view((-1, 20 * 24))
        return self.fc(x)
