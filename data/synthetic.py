import numpy as np
from numpy import typing as npt


class SineWave():
    """
    We'll use a sine wave as the input sequence and try to predict the next value in the sequence. 
    We'll create a dataset of data_size sequences with timesteps time steps each.

    timesteps determines the length of each input sequence, 
    data_size determines the number of samples we want to generate.
    """
    def __init__(self, timesteps: int = 10, data_size: int = 1000):
        self.timesteps = timesteps
        self.data_size = data_size
        
    def generate_sine(self) ->  tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generate Sine Wave.
        input_data is a 3-dimensional array with shape (data_size, timesteps, 1). 
        This means that we have data_size samples, each with timesteps time steps, and 1 feature at each time step. 
        output_data is a 2-dimensional array with shape (data_size, 1), which means that we have 
        data_size output samples, each with 1 feature.

        We generate the input sequence by calling np.linspace to create a sequence of timesteps values 
        evenly spaced between 0 and 3π (inclusive), and adding the random offset to each value. 
        We then pass this sequence through the np.sin function to generate the sine wave.

        We generate the output value by computing the sine of 3π plus the random offset. 
        This is the next value in the sine wave after the input sequence ends.
        """

        input_data = np.zeros((self.data_size, self.timesteps, 1))
        output_data = np.zeros((self.data_size, 1))

        for i in range(self.data_size):
            rand_offset = np.random.random() * 2 * np.pi
            input_data[i, :, 0] = np.sin(np.linspace(0.0 + rand_offset, 3 * np.pi + rand_offset, num=self.timesteps))
            output_data[i, 0] = np.sin(3 * np.pi + rand_offset)
        
        return input_data, output_data
