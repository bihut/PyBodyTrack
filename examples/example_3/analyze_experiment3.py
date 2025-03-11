import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def get_json_files(directory):
    """
    Recursively searches for all JSON files in the given directory.

    :param directory: Path to the directory to search.
    :return: List of JSON file paths.
    """
    json_files = []

    for root, _, files in os.walk(directory):  # Recursively walk through directories
        for file in files:
            if file.endswith(".json"):  # Check if it's a JSON file
                json_files.append(os.path.join(root, file))  # Store full path

    return json_files


def plot_movement_scatter(json_file):
    """Plots movement values in a scatter plot from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    times = [entry['time'] for entry in data]
    movements = [entry['movement'] for entry in data]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, movements, color='red', alpha=0.7)
    plt.xlabel("Frames")
    plt.ylabel("Movement")
    #plt.title("Scatter Plot of Movement Over Time")
    plt.grid(True)
    plt.savefig('movement_scatter.png', dpi=600, bbox_inches='tight')
    plt.show()


def convert_json_to_frame_paths(json_files):
    """
    Given a list of JSON file paths, replace 'output_cheby.json' with 'frame.jpg'.
    """
    return [file.replace("output_cheby.json", "frame.jpg") for file in json_files]
# Example usage:
# json_files = ["file1.json", "file2.json", "file3.json"]
# plot_exercise_metrics(json_files, "ram")

#delete_exercises = ["Sumo Deadlift","Leg Press Machine", "Romanian Deadlift"]
#keep_exercises = ["Plank","Leg Extension","Bicep Curl","Skull Crushers DB",
 #                 "Bench Press","Chins Up", "Military Press","Squat"]
path_file = "/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_3/video_output_euclidean.json"
plot_movement_scatter(path_file)
