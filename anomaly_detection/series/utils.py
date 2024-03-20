"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_event_timeline(event_log, color_by='event_type'):
    """
    Plots the event log as a timeline.

    Parameters:
        event_log (EventLogSeries): The event log series to plot.
        color_by (str): Column name to color-code events. Default is 'event_type'.
    """
    # Ensure there's a column to color by
    if color_by not in event_log.data.columns:
        print(f"Column '{color_by}' not found in event log data. Defaulting to no color coding.")
        color_by = None

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Generate colors if color coding is used
    colors = None
    if color_by:
        unique_vals = event_log.data[color_by].unique()
        colors = {val: plt.cm.tab10(i) for i, val in enumerate(unique_vals)}
        color_vals = [colors[val] for val in event_log.data[color_by]]
    else:
        color_vals = 'blue'  # Default color

    # Plot each event as a line on the timeline
    for i, (idx, row) in enumerate(event_log.data.iterrows()):
        ax.plot([idx, idx], [0, 1], color=color_vals[i] if color_by else 'blue', marker='o', markersize=8)
    
    # Improve layout
    ax.yaxis.set_visible(False)  # Hide the y-axis
    ax.spines['left'].set_visible(False)  # Hide the left spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['top'].set_visible(False)  # Hide the top spine
    
    # Set x-axis major locator to hour and formatter to auto DateFormatter
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Automatically format date labels in a readable way
    fig.autofmt_xdate()

    # Add legend if color coding is used
    if color_by and colors:
        patches = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8, label=val) for val, color in colors.items()]
        ax.legend(handles=patches)

    plt.title('Event Timeline')
    plt.tight_layout()
    plt.show()
