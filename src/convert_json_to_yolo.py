import json
import os
import argparse
import getpass

username = getpass.getuser()


def load_valid_classes(classes_file):
    """
    Reads the valid classes from classes.txt and returns a dictionary mapping old IDs to new YOLO-compatible IDs.
    """
    try:
        with open(classes_file, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: '{classes_file}' not found.")
        return {}

    num_classes = int(lines[0])  # First line is the number of classes
    valid_classes = {name: idx for idx, name in enumerate(lines[1:num_classes+1])}

    return valid_classes

def convert_json_to_yolo(json_file, output_dir, valid_classes, img_num=None):
    """
    Converts a JSON file in COCO format to YOLO format, filtering only valid classes.

    Args:
        json_file (str): Path to the input JSON file.
        output_dir (str): Directory to save the YOLO label files.
        valid_classes (dict): Mapping of class names to new YOLO-compatible indices.
        img_num (int, optional): Number of images to process. If None, processes all images.
    """

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
        return []

    categories = {cat['id']: cat['name'] for cat in data['categories']}

    # Organize annotations by image_id
    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_name = categories.get(ann['category_id'], None)
        
        # Skip annotations that are not in the valid class list
        if category_name not in valid_classes:
            continue

        if image_id not in image_annotations:
            image_annotations[image_id] = []
        
        # Remap class ID to match `classes.txt` order
        ann['category_id'] = valid_classes[category_name]
        image_annotations[image_id].append(ann)

    # Determine how many images to process
    image_ids = sorted(image_annotations.keys())  # Ensure a sorted order
    if img_num is not None:
        image_ids = image_ids[:img_num]  # Limit to specified number of images

    os.makedirs(output_dir, exist_ok=True)
    #os.chmod(output_dir, 0o777)
    
    # Process images and rename labels in FLIR_XXXXX format
    temp_filenames = []  # Store generated filenames for renaming later
    for i, image_id in enumerate(image_ids, start=1):  # Start numbering from 1
        yolo_lines = []
        for ann in image_annotations[image_id]:
            category_id = ann['category_id']
            bbox = ann['bbox']

            # Assuming image dimensions are 640x480 (adjust if needed)
            image_width = 640
            image_height = 480

            x_center = (bbox[0] + bbox[2] / 2) / image_width
            y_center = (bbox[1] + bbox[3] / 2) / image_height
            width = bbox[2] / image_width
            height = bbox[3] / image_height

            yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)

        # New FLIR_XXXXX filename format (5-digit zero-padded)
        temp_filename = f"FLIR_{str(i).zfill(5)}.txt"
        output_file = os.path.join(output_dir, temp_filename)
        temp_filenames.append(temp_filename)

        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"✅ YOLO labels for image {image_id} saved as {output_file}")

    return temp_filenames  # Return the generated filenames for renaming

def rename_yolo_files(output_dir, image_dir, temp_filenames):
    """
    Renames YOLO label files to match image filenames in the thermal_8_bit/ folder.

    Args:
        output_dir (str): Directory containing YOLO label files.
        image_dir (str): Directory containing image files.
        temp_filenames (list): List of temporary YOLO filenames to rename.
    """

    # Get sorted image filenames (without extensions)
    image_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    )

    # Ensure we rename only as many files as exist
    num_files_to_rename = min(len(temp_filenames), len(image_filenames))

    for i in range(num_files_to_rename):
        old_name = os.path.join(output_dir, temp_filenames[i])
        new_name = os.path.join(output_dir, f"{image_filenames[i]}.txt")

        os.rename(old_name, new_name)
        print(f"🔄 Renamed {old_name} → {new_name}")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format with class filtering.")
parser.add_argument("--img_num", type=int, default=None, help="Number of images to process. Default: all images.")

args = parser.parse_args()

# Define dataset paths
dataset_root = "/scratch/sfberrio/FLIR_ADAS_1_3"
splits = ["train", "val"]
classes_file = os.path.join('.', "classes.txt")

# Load valid class mappings
valid_classes = load_valid_classes(classes_file)
if not valid_classes:
    print("⚠️ No valid classes found. Please check 'classes.txt'. Exiting.")
    exit()

# Process both train and val annotations
for split in splits:
    json_file_path = os.path.join(dataset_root, split, "thermal_annotations.json")
    image_dir = os.path.join(dataset_root, split, "thermal_8_bit")  # Image directory
    output_dir = os.path.join(dataset_root, split, f"yolo_labels_{username}")  # YOLO labels directory

    print(f"📂 Processing {split} annotations: {json_file_path} -> {output_dir}")
    temp_filenames = convert_json_to_yolo(json_file_path, output_dir, valid_classes, args.img_num)

    # Rename generated YOLO files to match the image filenames
    rename_yolo_files(output_dir, image_dir, temp_filenames)

print("🎉 Conversion and renaming complete for both train and val datasets!")

