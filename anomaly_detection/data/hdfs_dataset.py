"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
from torch.utils.data import Dataset

from anomaly_detection.data.hdfs_series import HDFSEvents


class HDFSEventsDataset(Dataset):
    """
    Dataset class for training a model on HDFSEvents data.
    Each sample in the dataset consists of a context window of events and the target event.
    """

    def __init__(
        self,
        events: HDFSEvents,
        window_size: int = 10,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the HDFSEventsDataset.

        Parameters:
            events (HDFSEvents): The HDFSEvents object containing the event data.
            window_size (int): The size of the context window.
        """
        self.events = events
        self.window_size = window_size
        self.device = device

        # Prepare data for sliding window
        self.data = self._prepare_data()

    def _prepare_data(self) -> list:
        """
        Prepare data for sliding window.

        Returns:
            list: A list of tuples, where each tuple contains a context window and the target event.
        """
        data = []
        events_data = self.events.data

        # Iterate over each machine group
        for machine_id, machine_group in events_data.groupby("machine"):
            events_array = machine_group["event"].values
            # Create context windows
            for i in range(len(events_array) - self.window_size):
                context_window = torch.tensor(
                    events_array[i : i + self.window_size],
                    dtype=torch.long,
                    device=self.device,
                )
                target_event = torch.tensor(
                    events_array[i + self.window_size],
                    dtype=torch.long,
                    device=self.device,
                )
                data.append((context_window, target_event))

        return data

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset by index."""
        context_window, target_event = self.data[idx]
        return context_window, target_event
