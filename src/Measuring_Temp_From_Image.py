import numpy as np
import cv2
from pathlib import Path

# Define platform-independent file paths
tiff_file_path = Path("../data/FLIR_00001_16bit.tiff")
output_dir = Path("../runs")  # Directory to save the output images
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Fix OpenCV threading issue
cv2.ocl.setUseOpenCL(False)

# Read the 16-bit TIFF image
gray16_image = cv2.imread(str(tiff_file_path), cv2.IMREAD_ANYDEPTH)

# Check if the image was loaded successfully
if gray16_image is None:
    raise FileNotFoundError(f"Error: Could not read the file {tiff_file_path}")

print(f"Loaded image: {tiff_file_path} with shape {gray16_image.shape} and dtype {gray16_image.dtype}")

# Define pixel coordinates (adjust as needed)
x, y = 250, 250

# Access the pixel value at the specified coordinates
pixel_value = gray16_image[y, x]

# Convert pixel intensity to temperature in Celsius and Fahrenheit
if pixel_value > 0:  # Avoid invalid or negative readings
    temp_celsius = (pixel_value / 100) - 273.15
    temp_fahrenheit = (temp_celsius * 9 / 5) + 32
    temp_text = f"{temp_fahrenheit:.1f}°F"
else:
    temp_text = "Invalid Reading"

# Convert 16-bit image to 8-bit grayscale
gray8_image = cv2.normalize(gray16_image, None, 0, 255, cv2.NORM_MINMAX)
gray8_image = np.uint8(gray8_image)

# Mark the selected pixel in both grayscale images
cv2.circle(gray8_image, (x, y), 2, (0, 0, 0), -1)
cv2.circle(gray16_image, (x, y), 2, (0, 0, 0), -1)

# Annotate the temperature value on both images
cv2.putText(gray8_image, temp_text, (x - 80, y - 15),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
cv2.putText(gray16_image, temp_text, (x - 80, y - 15),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

# Save grayscale images (both 8-bit and 16-bit)
output_gray8 = output_dir / "FLIR_gray8_output.png"
output_gray16 = output_dir / "FLIR_gray16_output.png"

cv2.imwrite(str(output_gray8), gray8_image)
cv2.imwrite(str(output_gray16), gray16_image)

print(f"Saved 8-bit grayscale image to: {output_gray8}")
print(f"Saved 16-bit grayscale image to: {output_gray16}")
print(f"Temperature at ({x}, {y}): {temp_text}")
