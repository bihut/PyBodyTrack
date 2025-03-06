import os
import json
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


path_files = "/home/bihut/Documentos/UGR/Papers/pyBodyTrack-SoftwareX/ExperimentosVideos/Experimento1/output"
files = get_json_files(path_files)

files2 = sort_json_files_by_attribute(files, "nmi")
print("LEN FILES 2",len(files2))
print("FILES ORDENADOS:",files2)
