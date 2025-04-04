import getpass

# Get current username
username = getpass.getuser()

# Define base dataset root
base_path = f"/scratch/sfberrio/FLIR_ADAS_1_3"

# Construct paths with _username in the appropriate folders
train_images = f"{base_path}/train/thermal_8_bit_mini_{username}/images"
val_images = f"{base_path}/val/thermal_8_bit_mini_{username}/images"
train_labels = f"{base_path}/train/yolo_labels_{username}"
val_labels = f"{base_path}/val/yolo_labels_{username}"

# YAML content
yaml_content = f"""path: "{base_path}"  # Root directory relative to this YAML file

# Paths to images
train: "{train_images}"  # Path to training images
val: "{val_images}"  # Path to validation images

# Paths to labels
labels: "{train_labels}"  # Path to training labels
val_labels: "{val_labels}"  # Path to validation labels

dataset_name: FLIR_ADAS_1_3
dataset_type: object_detection

nc: 4  # Number of classes
names: ['person', 'car', 'bicycle', 'dog']  # Class names (corrected 'dogs' → 'dog' for consistency)
"""

# Write to file
with open("thermal_image_dataset.yaml", "w") as f:
    f.write(yaml_content)

print("✅ dataset_config.yaml generated successfully!")

