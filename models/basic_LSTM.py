import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Basic model with one LSTM layer having "hidden_size" hidden units and one fully connected output layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In the forward method, we pass the input x through the LSTM layer,
        take the output of the last time step, and pass it through the fully connected output layer.
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
