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

def sort_json_files_by_attribute(json_files, attribute):
    """
    Reads a list of JSON files, extracts the given attribute value,
    and returns the files sorted in ascending order of that attribute.

    :param json_files: List of JSON file paths.
    :param attribute: The JSON attribute to sort by.
    :return: List of tuples (file_path, attribute_value) sorted by the attribute.
    """
    files_with_attr = []

    for file_path in json_files:
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if attribute in data:  # Ensure the attribute exists in the JSON structure
                    files_with_attr.append((file_path, data[attribute]))
                else:
                    print(f"⚠️ Warning: '{attribute}' key not found in {file_path}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

    # Sort the list by the specified attribute value
    sorted_files = sorted(files_with_attr, key=lambda x: x[1])

    return [file[0] for file in sorted_files]  # Return only file paths


def plot_exercise_metrics_scatter(json_files, metric):
    data = []

    # Read data from JSON files
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    # Extract exercise names and metric values
    exercises = [entry['exercise'] for entry in data]
    values = [entry[metric] for entry in data]

    # Configure scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(exercises, values, color='b', alpha=0.7)

    # Set labels and title
    ax.set_xlabel("Exercise")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison of {metric} indicator by exercise")
    ax.set_xticklabels(exercises, rotation=45, ha="right")

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_exercise_metrics_all(json_files):
    data = []

    # Read data from JSON files
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    # Extract exercise names and all metric values
    exercises = [entry['exercise'] for entry in data]
    metrics = ['ram', 'nmi', 'mol', 'mof', 'mos']

    # Extract values for each metric
    values = {metric: [entry[metric] for entry in data] for metric in metrics}

    # Configure grouped bar chart
    x = np.arange(len(exercises))  # Positions for bars
    width = 0.15  # Width of each bar

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric)

    # Configure labels and title
    ax.set_xlabel("Exercise")
    ax.set_ylabel("Value")
    ax.set_title("Comparison of multiple indicators by exercise")
    ax.set_xticks(x + width * (len(metrics) / 2))
    ax.set_xticklabels(exercises, rotation=45, ha="right")
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


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
    bars = ax.bar(exercises, values, color='#beffc7', alpha=0.7)

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
    plt.savefig('/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_1/chart1.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_exercise_metrics_img(json_files, metric, images_dict=None):
    data = []

    # Read data from JSON files
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    # Extract exercise names and metric values
    exercises = [entry['exercise'] for entry in data]
    values = [entry[metric] for entry in data]

    # Configure bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(exercises))
    bars = ax.bar(x_positions, values, color='b', alpha=0.7)

    # Set labels and title
    ax.set_xlabel("Exercise")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison of {metric} indicator by exercise")
    ax.set_xticks(x_positions)  # Ensure xticks are properly set
    ax.set_xticklabels(exercises, rotation=45, ha="right")

    # Add images above each bar if provided
    if images_dict:
        for bar, exercise in zip(bars, exercises):
            if exercise in images_dict:
                image_path = images_dict[exercise]
                img = mpimg.imread(image_path)
                imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as needed
                ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    frameon=False, xycoords='data', xybox=(0, 10), boxcoords="offset points")
                ax.add_artist(ab)

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_exercise_metrics_img2(json_files, metric, images_dict=None):
    data = []

    # Read data from JSON files
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    # Extract exercise names and metric values
    exercises = [entry['exercise'] for entry in data]
    values = [entry[metric] for entry in data]

    # Configure figure with A4-like dimensions (29.7cm x 21cm), with 25x15cm for the chart
    fig = plt.figure(figsize=(29.7 / 2.54, 21 / 2.54))  # Convert cm to inches
    ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])  # Position for the 25x15cm chart

    x_positions = np.arange(len(exercises))
    bars = ax.bar(x_positions, values, color='b', alpha=0.7)

    # Set labels and title
    ax.set_xlabel("Exercise")
    ax.set_ylabel("Value")
    ax.set_title(f"Comparison of {metric} indicator by exercise")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(exercises, rotation=45, ha="right")

    # Add images around the plot if provided
    if images_dict:
        img_size = 0.1  # Image size relative to figure
        total_images = len(images_dict)

        left_x = 0.02
        right_x = 0.88
        top_y = 0.85
        bottom_y = 0.02

        left_images = total_images // 4
        right_images = total_images // 4
        top_images = total_images // 4
        bottom_images = total_images - (left_images + right_images + top_images)

        images_list = list(images_dict.items())

        # Distribute images
        for i, (exercise, image_path) in enumerate(images_list):
            img = mpimg.imread(image_path)
            imagebox = OffsetImage(img, zoom=img_size)

            if i < left_images:
                ab = AnnotationBbox(imagebox, (left_x, 0.15 + i * 0.6 / left_images), frameon=False,
                                    xycoords='figure fraction')
            elif i < left_images + right_images:
                ab = AnnotationBbox(imagebox, (right_x, 0.15 + (i - left_images) * 0.6 / right_images), frameon=False,
                                    xycoords='figure fraction')
            elif i < left_images + right_images + top_images:
                ab = AnnotationBbox(imagebox, (0.15 + (i - left_images - right_images) * 0.6 / top_images, top_y),
                                    frameon=False, xycoords='figure fraction')
            else:
                ab = AnnotationBbox(imagebox, (
                0.15 + (i - left_images - right_images - top_images) * 0.6 / bottom_images, bottom_y), frameon=False,
                                    xycoords='figure fraction')

            fig.add_artist(ab)

    # Show plot
    plt.show()


def convert_json_to_frame_paths(json_files):
    """
    Given a list of JSON file paths, replace 'output_cheby.json' with 'frame.jpg'.
    """
    return [file.replace("output_cheby.json", "frame.jpg") for file in json_files]
# Example usage:
# json_files = ["file1.json", "file2.json", "file3.json"]
# plot_exercise_metrics(json_files, "ram")

delete_exercises = ["Sumo Deadlift","Leg Press Machine", "Romanian Deadlift"]
keep_exercises = ["Plank","Leg Extension","Bicep Curl","Skull Crushers DB",
                  "Bench Press","Chins Up", "Military Press","Squat"]
path_files = "/home/bihut/Documentos/UGR/Papers/pyBodyTrack-SoftwareX/ExperimentosVideos/Experimento1/output"
files = get_json_files(path_files)
'''
for ex in delete_exercises:
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        if data['exercise'] == ex:
            files.remove(file)

'''
filesaux = []
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    if data['exercise'] in keep_exercises:
        print("Me quedo con ",data['exercise'])
        filesaux.append(file)

print(filesaux)
files=filesaux
files2 = sort_json_files_by_attribute(files, "nmi")
files3 = convert_json_to_frame_paths(files2)
dict = {}
for i in range(len(files2)):
    with open(files2[i], "r") as file:
        data = json.load(file)
    dict[data['exercise']] = files3[i]
#print(dict)
plot_exercise_metrics(files2,"nmi")
