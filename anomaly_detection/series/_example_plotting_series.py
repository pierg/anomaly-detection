"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import numpy as np
import pandas as pd

from anomaly_detection.series.events_time_series import EventLogSeries

# Sample event log data setup
data = {
    "timestamp": pd.date_range(start="2024-01-01", periods=24, freq="h"),
    "event_type": [
        "error",
        "warning",
        "info",
        "error",
        "info",
        "warning",
        "error",
        "info",
        "warning",
        "info",
        "error",
        "info",
        "warning",
        "error",
        "info",
        "warning",
        "info",
        "error",
        "info",
        "warning",
        "info",
        "error",
        "info",
        "warning",
    ],
    "severity": [
        3,
        2,
        1,
        3,
        1,
        2,
        3,
        1,
        2,
        1,
        3,
        1,
        2,
        3,
        1,
        2,
        1,
        3,
        1,
        2,
        1,
        3,
        1,
        2,
    ],
}

# Generate random hours to add
random_hours = np.random.randint(1, 100, size=24)
random_timestamps = pd.to_datetime("2024-01-01") + pd.to_timedelta(
    random_hours, unit="h"
)

# Sort the timestamps
random_timestamps_sorted = np.sort(random_timestamps)

# Update the data dictionary with sorted random timestamps
data["timestamp"] = random_timestamps_sorted

# Recreate the DataFrame with the corrected timestamps
df_random_corrected = pd.DataFrame(data)
df_random_corrected.set_index("timestamp", inplace=True)

# Update the data dictionary with random timestamps
data["timestamp"] = random_timestamps

df = pd.DataFrame(data)
# Ensure timestamp is datetime type
df["timestamp"] = pd.to_datetime(df["timestamp"])
# Set the timestamp as the DataFrame index
df.set_index("timestamp", inplace=True)

event_log = EventLogSeries(df)

event_log.visualize()
