import os
import shutil
import argparse
import re
import getpass

username = getpass.getuser()

# Manually adjustable train/val split ratio
TRAIN_SPLIT_RATIO = 0.6  # Change this value to control the split ratio

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def create_mini_dataset_split(source_dir, target_root, start_idx, img_num):
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    image_files = sorted(
        [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg'))],
        key=natural_sort_key
    )

    total_available = len(image_files)
    if start_idx >= total_available:
        print(f"⚠️ Start index {start_idx} is beyond the number of available images ({total_available}).")
        return

    selected_files = image_files[start_idx:start_idx + img_num] if img_num else image_files[start_idx:]

    num_selected = len(selected_files)
    if num_selected < img_num:
        print(f"⚠️ Requested {img_num} images but only {num_selected} are available from index {start_idx}.")

    train_cutoff = int(num_selected * TRAIN_SPLIT_RATIO)
    train_files = selected_files[:train_cutoff]
    val_files = selected_files[train_cutoff:]

    # Define output directories
    train_target_dir = os.path.join(target_root, "train", f"thermal_8_bit_mini_{username}", "images")
    val_target_dir = os.path.join(target_root, "val", f"thermal_8_bit_mini_{username}", "images")
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_target_dir, f))
        print(f"📥 Train: {f}")

    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_target_dir, f))
        print(f"📥 Val: {f}")

    print(f"✅ Successfully created split mini dataset: {len(train_files)} train / {len(val_files)} val\n")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Create a split mini dataset by copying a subset of JPEG images.")
parser.add_argument("--img_num", type=int, required=True, help="Number of images to copy.")
parser.add_argument("--start_idx", type=int, default=0, help="Start index for image selection. Default: 0")
args = parser.parse_args()

# Define dataset root path and source
dataset_root = "/scratch/sfberrio/FLIR_ADAS_1_3"
source_images_dir = os.path.join(dataset_root, "train", "thermal_8_bit")

print(f"📂 Processing mini dataset creation from {source_images_dir} starting at index {args.start_idx}")
print(f"📊 Train/Val split: {TRAIN_SPLIT_RATIO:.0%}/{(1 - TRAIN_SPLIT_RATIO):.0%}")

create_mini_dataset_split(source_images_dir, dataset_root, args.start_idx, args.img_num)

print("✅ Mini dataset creation complete!")

