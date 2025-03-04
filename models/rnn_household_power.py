import torch
from torch import nn


class RNN(nn.Module):
    """
    the class is initialized with input_size, hidden_size, and output_size.
    These parameters define the size of the input, hidden state, and output of the RNN.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method takes two arguments: x and hidden.
        Args:
            x: the input to the RNN, which is a tensor of shape (batch_size, seq_len, input_size).
            hidden: the initial hidden state of the RNN, which is a tensor of shape (1, batch_size, hidden_size).
        """
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
