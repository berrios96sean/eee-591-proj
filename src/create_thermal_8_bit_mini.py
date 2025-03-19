import os
import shutil
import argparse

def create_mini_dataset(source_dir, target_base_dir, img_num):
    """
    Creates a mini dataset by copying the first `img_num` JPEG images from source_dir
    to a 'images' subdirectory inside target_base_dir.

    Args:
        source_dir (str): Path to the original full dataset directory.
        target_base_dir (str): Path to the mini dataset base directory.
        img_num (int): Number of images to copy.
    """

    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Define the target images directory inside 'thermal_8_bit_mini'
    target_images_dir = os.path.join(target_base_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)  # Create the target directory if it doesn't exist

    # Get a sorted list of JPEG image files
    image_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

    # Check if there are enough images
    if len(image_files) < img_num:
        print(f"Warning: Requested {img_num} images, but only {len(image_files)} are available.")
        img_num = len(image_files)  # Adjust to available images

    # Copy images
    for i in range(img_num):
        src_path = os.path.join(source_dir, image_files[i])
        dest_path = os.path.join(target_images_dir, image_files[i])
        shutil.copy(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")

    print(f"✅ Successfully created mini dataset with {img_num} images in {target_images_dir}\n")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Create a mini dataset by copying a subset of JPEG images.")
parser.add_argument("--img_num", type=int, required=True, help="Number of images to copy.")

args = parser.parse_args()

# Define dataset paths
dataset_root = "/scratch/sfberrio/FLIR_ADAS_1_3"
splits = ["train", "val"]

for split in splits:
    source_images_dir = os.path.join(dataset_root, split, "thermal_8_bit")
    target_base_dir = os.path.join(dataset_root, split, "thermal_8_bit_mini")

    print(f"Processing {split}: {source_images_dir} -> {target_base_dir}/images")
    create_mini_dataset(source_images_dir, target_base_dir, args.img_num)

print("✅ Mini dataset creation complete for both train and val.")

