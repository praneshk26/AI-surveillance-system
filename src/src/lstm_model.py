import torch
import torch.nn as nn

class ActivityLSTM(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128, num_layers=2, output_dim=2):
        super(ActivityLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape = (batch, sequence_length, feature_dim)
        out, _ = self.lstm(x)
        
        # We use the last timestep output
        final_output = out[:, -1, :]
        
        logits = self.fc(final_output)
        prob = self.softmax(logits)
        
        return prob


# Test code
if __name__ == "__main__":
    model = ActivityLSTM()
    dummy_seq = torch.randn(1, 10, 128)  # batch=1, seq=10, features=128
    output = model(dummy_seq)
    print("Output probabilities:", output)


