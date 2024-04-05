"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import pandas as pd
import plotly.graph_objects as go

from anomaly_detection.series.base_time_series import BaseTimeSeries


class PointProcessTimeSeries(BaseTimeSeries):
    """
    Represents a time series of events occurring over time, modeled as a point process.
    Data is expected in a pandas DataFrame with a DateTimeIndex or a column that can be converted to DateTime.
    """

    def __init__(self, data: pd.DataFrame, time_column: str = None) -> None:
        """
        Initialize the point process time series object.

        Parameters:
            data (pd.DataFrame): The event data. Can be a DataFrame with a DateTimeIndex or contain a specific column with datetime values.
            time_column (str, optional): The name of the column that contains datetime values if not using a DateTimeIndex.
        """
        super().__init__(data)
        assert (
            "event_type" in self.data.columns
        ), "Data must include 'event_type' column."

        # If a time_column is provided, set it as the index
        if time_column is not None:
            self.data[time_column] = pd.to_datetime(self.data[time_column])
            self.data.set_index(time_column, inplace=True)
        else:
            # Ensure the index is of datetime type
            assert isinstance(
                self.data.index, pd.DatetimeIndex
            ), "Data must have a DateTimeIndex or provide a time column."

    def preprocess(self) -> None:
        """
        Preprocess the event data. This could involve sorting or removing duplicates.
        This implementation assumes the data is already sorted and uniquely timestamped.
        """
        # Example: Removing duplicate timestamps if necessary
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    def summarize(self) -> pd.DataFrame:
        """
        Summarize the point process by calculating inter-event times and providing basic statistics.

        Returns:
            pd.DataFrame: A DataFrame with inter-event times and their basic statistics.
        """
        if len(self.data) > 1:
            # Calculate inter-event times
            inter_event_times = self.data.index.to_series().diff().dropna()
            summary_stats = inter_event_times.describe()
            return summary_stats
        else:
            return pd.DataFrame()  # Return an empty DataFrame if not enough events

    def inter_event_times(self) -> pd.Series:
        """
        Calculate and return the inter-event times as a pandas Series.

        Returns:
            pd.Series: Series of inter-event times.
        """
        if len(self.data) > 1:
            return self.data.index.to_series().diff().dropna()
        else:
            return pd.Series(dtype="timedelta64[ns]")

    def visualize(self, title="Event Types Over Time"):
        # Assign alphanumeric labels to unique event types
        unique_types = self.data["event_type"].unique()
        type_labels = {
            etype: f"E{index}" for index, etype in enumerate(unique_types, start=1)
        }

        # Prepare the data
        self.data["label"] = self.data["event_type"].map(type_labels)

        # Create the figure
        fig = go.Figure()

        # Add dots for events
        for etype in unique_types:
            df_filtered = self.data[self.data["event_type"] == etype]
            fig.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=[1] * len(df_filtered),
                    mode="markers+lines",
                    name=etype,
                    text=df_filtered["label"],
                    marker=dict(size=8),
                    hoverinfo="text",
                )
            )

        # Improve layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis=dict(tickmode="array", tickvals=[1], ticktext=["Events"]),
            yaxis_showgrid=False,
            yaxis_zeroline=False,
        )

        fig.show()
