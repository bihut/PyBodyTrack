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



def plot_exercise_metrics(json_files, metric):
    data = []

    # Read data from JSON files
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    # Extract exercise names and metric values
    exercises = [entry['exercise'] for entry in data]
    values = [entry[metric] for entry in data]

    # Configure bar chart
    fig, ax = plt.subplots(figsize=(21 / 2.54, 15 / 2.54))
    bars = ax.bar(exercises, values, color='b', alpha=0.7)

    # Add value labels inside each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * max(values), f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    # Configure labels and title
    ax.set_ylabel("NMI (Normalized Motion Index)")  # Change 'Value' to 'NMI'
    #ax.set_title(f"Comparison of {metric} indicator by exercise")
    ax.set_xticklabels(exercises, rotation=45, ha="right", fontsize=12, fontweight='bold')  # Bold and larger font

    # Show plot
    plt.savefig('/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_4/chart4.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
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
path_files = "/home/bihut/Documentos/UGR/Papers/pyBodyTrack-SoftwareX/ExperimentosVideos/Experimento4/output"
files = get_json_files(path_files)
'''
for ex in delete_exercises:
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        if data['exercise'] == ex:
            files.remove(file)

'''

files2 = sort_json_files_by_attribute(files, "nmi")
files3 = convert_json_to_frame_paths(files2)
dict = {}
for i in range(len(files2)):
    with open(files2[i], "r") as file:
        data = json.load(file)
    dict[data['exercise']] = files3[i]
#print(dict)
plot_exercise_metrics(files2,"nmi")
