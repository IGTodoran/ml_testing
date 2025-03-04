from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import typing as npt
from torch import nn, optim

from data.household_electric_power.household_electric_power import (
    HouseholdElectricPower,
)
from models.rnn_household_power import RNN


def visualize_power_day(date1: str, dataset: pd.DataFrame) -> None:
    """
    Visualize power for a day
    """
    _ = dataset.loc[date1].plot(kind="line", y="Global_active_power", figsize=(10, 6), grid=True)
    plt.show()
    plt.close()


def create_sequences(in_data: pd.DataFrame, seq_len: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    data: numpy array
        The input time series data
    seq_len: int
        The length of the input sequence. Number of past time steps to use for prediction
    """

    # initialize empty lists
    X = []
    y = []
    for i in range(seq_len, len(in_data)):
        X.append(in_data[i - seq_len : i])
        y.append(in_data[i])
    return np.array(X), np.array(y)


def create_train_test_datasets(
    dataset: pd.DataFrame, p: float, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        dataset (str): Pandas dataframe containg the entire data
        p (int): Proportion of data to be used for training
        seq_len (int):  number of past time steps to use for prediction

    Returns:
        None
    """
    train_size = int(len(dataset) * p)
    train_data = dataset.iloc[:train_size].values
    test_data = dataset.iloc[train_size:].values

    # Create train and test sequences
    X_train, y_train = create_sequences(train_data, seq_len)
    X_test, y_test = create_sequences(test_data, seq_len)

    # convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def create_optimizer(model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
    """
    Create optimizer for training RNN.
    """
    return optim.Adam(model.parameters(), lr=lr)


def train_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, num_epochs: int, hidden_size: int, learning_rate: float
) -> nn.Module:
    """
    Loop through each epoch and train the RNN model.
    """
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model=model, lr=learning_rate)

    for epoch in range(num_epochs):
        # set the initial hidden state
        hidden = torch.zeros(1, X.size(0), hidden_size)

        # forward pass
        outputs, hidden = model(X, hidden)
        loss = criterion(outputs, y)

        # backwards and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss at every 10th epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {round(loss.item(), 4)}")

    return model


def evaluate_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, hidden_size: int, max_power: float, min_power: float
):
    """
    Evaluate the trained model.
    """
    # Set initial hidden state for test data
    hidden = torch.zeros(1, X.size(0), hidden_size)

    # Forward pass
    test_outputs, _ = model(X, hidden)

    # Inverse normalize the output and inputs
    test_outputs = (test_outputs * (max_power - min_power)) + min_power
    y_test = (y * (max_power - min_power)) + min_power

    # Compute the test loss
    criterion = nn.MSELoss()
    test_loss = criterion(test_outputs, y_test)

    print(f"Test Loss: {round(test_loss.item(), 4)}")

    # Convert the output and labels to numpy arrays
    test_outputs = test_outputs.detach().numpy()
    y_test = y_test.numpy()
    # Plot the first 100 actual and predicted values
    plt.plot(y_test[200:300], label="actual")
    plt.plot(test_outputs[200:300], label="predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = Path("./data/household_electric_power/household_power_consumption.txt")

    raw_data = HouseholdElectricPower(data_path=data_path)
    data, max_power, min_power = raw_data.preprocess_data()

    data.tail()
    visualize_power_day(date1="2009-05-08", dataset=data)

    input_size = 1  # number of features in the input
    hidden_size = 32  # number of hidden units in the RNN layer
    output_size = 1  # number of output features
    learning_rate = 0.001
    num_epochs = 100
    seq_len = 6

    model = RNN(input_size, hidden_size, output_size)

    X_train, y_train, X_test, y_test = create_train_test_datasets(dataset=data, p=0.8, seq_len=seq_len)
    trained_model = train_model(
        model=model,
        X=X_train,
        y=y_train,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    evaluate_model(
        model=trained_model, X=X_test, y=y_test, hidden_size=hidden_size, max_power=max_power, min_power=min_power
    )
