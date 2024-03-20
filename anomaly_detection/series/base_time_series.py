"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from abc import ABC, abstractmethod
import plotly.express as px
import pandas as pd
import numpy as np
import torch

class BaseTimeSeries(ABC):
    """
    Abstract base class for a time series stored in a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the time series object with data stored in a pandas DataFrame.

        Parameters:
            data (pd.DataFrame): The time series data.
        """
        self.data = data

    @abstractmethod
    def preprocess(self) -> None:
        """
        Preprocess the data. This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def summarize(self) -> None:
        """
        Summarize the time series data. This method should return a summary suitable for the data type.
        """
        pass

    def to_numpy(self) -> np.ndarray:
        """
        Export the time series data to a Numpy array.

        Returns:
            np.ndarray: The data as a Numpy array.
        """
        return self.data.values

    def to_tensor(self) -> torch.Tensor:
        """
        Export the time series data to a PyTorch tensor.

        Returns:
            torch.Tensor: The data as a PyTorch tensor.
        """
        return torch.tensor(self.data.values, dtype=torch.float)
    
    def visualize(self, title="Time Series Visualization", y=None, **kwargs):
        """
        Visualizes the time series data using Plotly.

        Parameters:
            title (str): The title of the plot.
            y (str | list of str, optional): The column(s) to be plotted on the y-axis. Defaults to the first column.
            **kwargs: Additional keyword arguments for Plotly's line plot function.
        """
        if y is None:
            y = self.data.columns[0]  # Default to the first column if none specified
        fig = px.line(self.data.reset_index(), x=self.data.index.name, y=y, title=title, **kwargs)
        fig.show()
