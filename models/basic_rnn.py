"""
Basic RNN.
"""

from torch import nn


class RNN(nn.Module):
    """
    Basic RNN
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, batch_first: bool = True):
        super(RNN, self).__init__()

        # Create a new RNN layer with 1 input feature, 32 hidden units, and 1 layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

        # Create a new fully connected layer with 32 input features and 1 output feature
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        """
        Passes the input tensor through the RNN and the fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Pass the input tensor through the RNN layer, which returns a new tensor with
        # shape (batch_size, sequence_length, hidden_size)
        rnn_out, _ = self.rnn(x)
        # Note that the _ indicates that we are only interested in the first output of self.rnn(x), 
        # which is the output tensor, and not the second output, which is the final hidden state of the RNN layer.
        # Since we are not using this hidden state, we can ignore it by assigning it to _.

        # Pass the last output from the RNN layer through the fully connected layer,
        # which returns a new tensor with shape (batch_size, 1)
        fc_out = self.fc(rnn_out[:, -1, :])

        # Return the output tensor
        return fc_out
