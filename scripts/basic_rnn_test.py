import matplotlib.pyplot as plt
import numpy as np
import torch

from numpy import typing as npt
from torch import nn
from torch import optim

from data.synthetic import SineWave
from models.basic_rnn import RNN


def visualize_sine(input_data: npt.NDArray[np.float64], output_data: npt.NDArray[np.float64]) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for i in range(2):
        for j in range(3):
            # plot input sequence
            axs[i, j].plot(input_data[i + j, :, 0], label='Input')

            # plot output value with big marker
            axs[i, j].plot(range(timesteps - 1, timesteps), output_data[i + j], marker='o', markersize=10, label='Output')

            # set plot title, axis labels, and legend
            axs[i, j].set_title(f'Sample {i+j+1}')
            axs[i, j].set_xlabel('Time Step')
            axs[i, j].set_ylabel('Feature Value / Value')
            axs[i, j].legend()

    plt.suptitle('Input and Output Sequences')
    plt.tight_layout()
    plt.show()


def create_train_test_datasets(
        data_size: int,
        input_data: npt.NDArray[np.float64],
        output_data: npt.NDArray[np.float64],
        p=0.8) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create train and test datasets.
    Train dataset will be int(data_size * p)
    """
    train_size = int(data_size * p)

    X_train = input_data[:train_size, :, :]
    y_train = output_data[:train_size, :]

    X_test = input_data[train_size:, :, :]
    y_test = output_data[train_size:, :]

    return X_train, y_train, X_test, y_test


def create_optimizer(model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
    """
    Create optimizer for training RNN.
    """
    return optim.Adam(model.parameters(), lr=lr)


def train_rnn(rnn_model: nn.Module) -> tuple[list, list]:
    """
    Training of RNN.
    """
    train_losses = []
    val_losses = []

    criterion = nn.MSELoss()
    optimizer = create_optimizer(model=rnn, lr=0.01)

    for epoch in range(70):
        rnn_model.train()
        optimizer.zero_grad()

        outputs = rnn_model(torch.Tensor(X_train))
        loss = criterion(outputs, torch.Tensor(y_train))

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        rnn_model.eval()
        with torch.no_grad():
            outputs = rnn_model(torch.Tensor(X_test))
            val_loss = criterion(outputs, torch.Tensor(y_test))
            val_losses.append(val_loss.item())

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, 70, loss.item(), val_loss.item()))
    
    return train_losses, val_losses


timesteps = 10
data_size = 1000

gen_sine = SineWave(timesteps=timesteps, data_size=data_size)
input_data, output_data = gen_sine.generate_sine()

# visualize_sine(input_data=input_data, output_data=output_data)
X_train, y_train, X_test, y_test = create_train_test_datasets(data_size=data_size, input_data=input_data, output_data=output_data)

input_size = 1
hidden_size = 32
num_layers = 1
batch_first = True
rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

train_loss, val_loss = train_rnn(rnn_model=rnn)

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()
