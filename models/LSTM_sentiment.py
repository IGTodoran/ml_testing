import torch
from torch import nn


class SimpleLSTM(nn.Module):
    """
    We define a simple LSTM model class that inherits from nn.Module.
    The model consists of an embedding layer, an LSTM layer, and a fully connected (linear) layer.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        The forward method takes an input tensor x, passes it through the embedding layer, the LSTM layer,
        and finally the fully connected layer to produce the output logits.
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.fc(hidden.squeeze(0))
        return logits
