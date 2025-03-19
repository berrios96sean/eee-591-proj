import json

def convert_json_to_yolo(json_file, output_dir="yolo_labels"):
    """
    Converts a JSON file in COCO format to YOLO format for 20 images.

    Args:
        json_file (str): Path to the input JSON file.
        output_dir (str): Directory to save the YOLO label files.
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

    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id < 20: #Only process the first 20 images
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

    import os
    os.makedirs(output_dir, exist_ok=True)

    for image_id, annotations in image_annotations.items():
        yolo_lines = []
        for ann in annotations:
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

        output_file = os.path.join(output_dir, f"{image_id}.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        print(f"YOLO labels for image {image_id} saved to {output_file}")

json_file_path = r"/scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_annotations.json"
convert_json_to_yolo(json_file_path)
