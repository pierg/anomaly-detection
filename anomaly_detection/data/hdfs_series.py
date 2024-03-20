"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from pathlib import Path
import numpy as np
import pandas as pd
from series.events_time_series import LogEventTimeSeries

class HDFSEvents(LogEventTimeSeries):
    """
    Represents a sequence of HDFS events with timestamp information and machine IDs.
    Inherits from LogEventTimeSeries.
    **Format**: The assumed format of a text file is that each line in
            the text file contains a space-separated sequence of event IDs for a
            machine. I.e. for *n* machines, there will be *n* lines in the file.
    """
    
    def __init__(self, data: pd.DataFrame, event_column: str, time_column: str, machine_column: str) -> None:
        """
        Initialize the HDFSEvents object.

        Parameters:
            data (pd.DataFrame): The event data with timestamps and machine IDs.
            event_column (str): The name of the column that contains the event information.
            time_column (str): The name of the column that contains timestamp information.
            machine_column (str): The name of the column that contains machine information.
        """
        super().__init__(data, event_column, time_column)
        self.machine_column = machine_column

    @classmethod
    def from_text_file(cls, path: Path, nrows: int = None) -> 'HDFSEvents':
        """
        Create an HDFSEvents object from a text file.

        Parameters:
            path (Path): The path to the text file.
            nrows (int, optional): Number of rows to read from the file.

        Returns:
            HDFSEvents: The HDFSEvents object.
        """
        events = []
        machines = []

        with open(path) as infile:
            for machine, line in enumerate(infile):
                if nrows is not None and machine >= nrows:
                    break
                for event in map(int, line.split()):
                    events.append(event)
                    machines.append(machine)

        data = pd.DataFrame({
            'timestamp': np.arange(len(events)),  # Increasing order
            'event': events,
            'machine': machines,
        })
        
        return cls(data, event_column='event', time_column='timestamp', machine_column='machine')


