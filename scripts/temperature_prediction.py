"""
Predict temperature using LSTM
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from data.temperature.temperature import TemperatureDataset, create_data_sequences
from models.basic_LSTM import LSTM


def create_optimizer(model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
    """
    Create optimizer for training RNN.
    """
    return optim.Adam(model.parameters(), lr=lr)


def train_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, num_epochs: int, learning_rate: float, batch_size: int
) -> nn.Module:
    """
    Loop through each epoch and train the RNN model.
    """
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model=model, lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Shuffle the training data
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        y = y[perm]

        # Loop over batches
        for i in range(0, X.shape[0], batch_size):
            # Get batch
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print loss for this epoch
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item(): .4f}")
    return model


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the trained model.
    """
    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    # Calculate the test loss
    criterion = nn.MSELoss()
    test_loss = criterion(y_pred, y)
    print(f"Test Loss: {test_loss.item(): .4f}")

    return y_pred


def visualize_prediction(y_test: torch.Tensor, y_pred: torch.Tensor) -> None:
    """
    Visualize the predictions and the true values (test dataset).
    """
    # Convert Pytorch tensors to numpy arrays
    y_test_np = y_test.numpy()
    y_pred_np = y_pred.numpy()

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_np[:500], label="Actual")
    plt.plot(y_pred_np[:500], label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("LSTM Predictions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = Path("./data/temperature/temperature.csv")
    temperature_dataset = TemperatureDataset(data_path=data_path)
    train_data, test_data = temperature_dataset.split_dataset(p=0.7)

    # Create sequences for training and testing data
    seq_length = 24
    X_train, y_train = create_data_sequences(data=train_data, seq_length=seq_length)
    X_test, y_test = create_data_sequences(data=test_data, seq_length=seq_length)

    # Instantiate the model
    input_size = X_train.shape[2]
    hidden_size = 32
    output_size = 1
    model = LSTM(input_size, hidden_size, output_size)

    # Convert numpy arrays to Pytorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Define the batch size and number of epochs
    batch_size = 32
    num_epochs = 25
    lr = 0.001
    trained_model = train_model(
        model=model, X=X_train, y=y_train, num_epochs=num_epochs, learning_rate=lr, batch_size=batch_size
    )

    y_pred = evaluate_model(model=trained_model, X=X_test, y=y_test)

    visualize_prediction(y_test=y_test, y_pred=y_pred)
