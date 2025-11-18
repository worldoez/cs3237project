import torch, torch.nn as nn

class LSTMIMU(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2, num_classes=4, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Dropout(0.2),
            nn.Linear(out_size, num_classes)
        )

    def forward(self, x):  # x: (B,T,C)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (B,H)
        logits = self.head(last)
        return logits