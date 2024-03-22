"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from anomaly_detection.utils.paths import output_folder
import json

def save_results(config_id, precision, recall, f1_score):
    # Define the function for saving results
    results_file = output_folder / "results.json"
    if not results_file.exists():
        with open(results_file, 'w') as f:
            json.dump({}, f)

    with open(results_file, 'r+') as f:
        results = json.load(f)
        results[config_id] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
        f.seek(0)
        json.dump(results, f, indent=4)
        f.truncate()
