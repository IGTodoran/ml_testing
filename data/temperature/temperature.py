from pathlib import Path

import numpy as np
import pandas as pd
from numpy import typing as npt


class TemperatureDataset:
    """
    Sample dataset that contains temperature readings for a single sensor over time.
    We'll load the dataset into a pandas DataFrame and preprocess it so that it can be fed into our LSTM model.
    """

    def __init__(self, data_path: Path):
        self.df = pd.read_csv(data_path)

    def preprocess(self) -> npt.NDArray[np.float64]:
        """
        Run a serie of transformations for preparing the dataset.
        """

        # Convert the 'datetime' column to a datetime object
        self.df["Date"] = pd.to_datetime(self.df["Date"])

        # Set the 'datetime' column as the index
        self.df.set_index("Date", inplace=True)

        # Resample the data to hourly intervals and fill missing values with the previous value
        self.df = self.df.resample("h").ffill()

        # Normalize the data
        self.df = (self.df - self.df.mean()) / self.df.std()

        # Convert the DataFrame to a numpy array
        return self.df.values

    def split_dataset(self, p: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Split the data into training and testing sets.
        We'll use the first p% of the data for training and the remaining (1-p)% for testing.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p parameters needs to be between 0 and 1")
        data = self.preprocess()
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.7)
        return data[:train_size], data[train_size:]


def create_data_sequences(
    data: npt.NDArray[np.float64], seq_length: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Before we can train our LSTM model, we need to create sequences of data that the model can learn from.
    We'll create sequences of length seq_length hours, and we'll use a sliding window approach to create overlapping sequences.
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
