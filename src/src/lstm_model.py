import torch
import torch.nn as nn

class ActivityLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, output_dim=2):
        super(ActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = ActivityLSTM()
    sample = torch.randn(1, 10, 256)
    output = model(sample)
    print("Output:", output)

