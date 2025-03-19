import json
import os
import argparse

def convert_json_to_yolo(json_file, output_dir, img_num=None):
    """
    Converts a JSON file in COCO format to YOLO format.

    Args:
        json_file (str): Path to the input JSON file.
        output_dir (str): Directory to save the YOLO label files.
        img_num (int, optional): Number of images to process. If None, processes all images.
    """

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
        return

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Organize annotations by image_id
    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Determine how many images to process
    image_ids = sorted(image_annotations.keys())  # Ensure a sorted order
    if img_num is not None:
        image_ids = image_ids[:img_num]  # Limit to specified number of images

    os.makedirs(output_dir, exist_ok=True)

    # Process images and rename labels in FLIR_XXXXX format
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

            yolo_line = f"{category_id - 1} {x_center} {y_center} {width} {height}"  # YOLO class indices start from 0
            yolo_lines.append(yolo_line)

        # New FLIR_XXXXX filename format (5-digit zero-padded)
        new_filename = f"FLIR_{str(i).zfill(5)}.txt"
        output_file = os.path.join(output_dir, new_filename)

        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"✅ YOLO labels for image {image_id} saved as {output_file}")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format.")
parser.add_argument("--img_num", type=int, default=None, help="Number of images to process. Default: all images.")

args = parser.parse_args()

# Define dataset paths
dataset_root = "/scratch/sfberrio/FLIR_ADAS_1_3"
splits = ["train", "val"]

# Process both train and val annotations
for split in splits:
    json_file_path = os.path.join(dataset_root, split, "thermal_annotations.json")
    output_dir = os.path.join(dataset_root, split, "yolo_labels")  # Change to yolo_labels directory

    print(f"📂 Processing {split} annotations: {json_file_path} -> {output_dir}")
    convert_json_to_yolo(json_file_path, output_dir, args.img_num)

print("🎉 Conversion complete for both train and val datasets!")

