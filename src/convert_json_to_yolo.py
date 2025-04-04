import json
import os
import argparse
import getpass
import math

username = getpass.getuser()

# Manually adjustable train/val split ratio
TRAIN_SPLIT_RATIO = 0.6  # Change this value (0.0 to 1.0) to control the split ratio

def load_valid_classes(classes_file):
    try:
        with open(classes_file, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: '{classes_file}' not found.")
        return {}

    num_classes = int(lines[0])
    valid_classes = {name: idx for idx, name in enumerate(lines[1:num_classes+1])}
    return valid_classes

def convert_json_to_yolo(json_file, output_dir, valid_classes, image_subset):
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
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_name = categories.get(ann['category_id'], None)
        if category_name not in valid_classes:
            continue

        if image_id not in image_annotations:
            image_annotations[image_id] = []
        ann['category_id'] = valid_classes[category_name]
        image_annotations[image_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)
    temp_filenames = []
    for image_id in image_subset:
        if image_id not in image_annotations:
            continue

        yolo_lines = []
        image_filename = image_id_to_filename[image_id]
        width = 640
        height = 480

        for img in data['images']:
            if img['id'] == image_id:
                width = img['width']
                height = img['height']
                break

        for ann in image_annotations[image_id]:
            category_id = ann['category_id']
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            box_width = bbox[2] / width
            box_height = bbox[3] / height

            yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_lines.append(yolo_line)

        output_filename = os.path.splitext(os.path.basename(image_filename))[0] + ".txt"
        output_file = os.path.join(output_dir, output_filename)
        temp_filenames.append((output_filename, image_id))

        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"✅ YOLO labels for image {image_id} saved as {output_file}")

    return temp_filenames

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format with class filtering.")
parser.add_argument("--img_num", type=int, default=None, help="Number of images to process. Default: all images.")
parser.add_argument("--start_idx", type=int, default=0, help="Start index for image processing. Default: 0.")
args = parser.parse_args()

# Set dataset paths
dataset_root = "/scratch/sfberrio/FLIR_ADAS_1_3"
split = "train"
classes_file = os.path.join('.', "classes.txt")

# Load classes
valid_classes = load_valid_classes(classes_file)
if not valid_classes:
    print("⚠️ No valid classes found. Please check 'classes.txt'. Exiting.")
    exit()

# Load annotation metadata
json_file_path = os.path.join(dataset_root, split, "thermal_annotations.json")
with open(json_file_path, 'r') as f:
    data = json.load(f)
image_list = sorted(data['images'], key=lambda x: x['file_name'])

total_images = len(image_list)
start = args.start_idx
end = min(start + args.img_num, total_images) if args.img_num else total_images
selected_images = image_list[start:end]

train_count = math.ceil(len(selected_images) * TRAIN_SPLIT_RATIO)
train_ids = [img['id'] for img in selected_images[:train_count]]
val_ids = [img['id'] for img in selected_images[train_count:]]

train_output_dir = os.path.join(dataset_root, "train", f"yolo_labels_{username}")
val_output_dir = os.path.join(dataset_root, "val", f"yolo_labels_{username}")

print(f"📂 Processing training annotations from {start} to {start+train_count-1} → {train_output_dir}")
train_temp = convert_json_to_yolo(json_file_path, train_output_dir, valid_classes, train_ids)

print(f"📂 Processing validation annotations from {start+train_count} to {end-1} → {val_output_dir}")
val_temp = convert_json_to_yolo(json_file_path, val_output_dir, valid_classes, val_ids)

print("🎉 60/40 split conversion complete!")

