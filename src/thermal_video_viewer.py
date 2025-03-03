import cv2
import numpy as np
import tifffile
from pathlib import Path

# Define a platform-independent file path
tiff_file_path = Path("../data/FLIR_video_00001.tiff")
output_dir = Path("../runs")  # Directory to save the output image
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

print(f"Processing: {tiff_file_path}")

x_mouse, y_mouse = 0, 0

def mouse_events(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global x_mouse, y_mouse
        x_mouse = x
        y_mouse = y

with tifffile.TiffFile(tiff_file_path) as tif:
    for idx, page in enumerate(tif.pages):
        gray16_frame = page.asarray()

        # Convert temperature value at the mouse pointer
        temperature_pointer = gray16_frame[y_mouse, x_mouse]
        temperature_pointer = (temperature_pointer / 100) - 273.15  # Convert to Celsius
        temperature_pointer = (temperature_pointer * 9 / 5) + 32  # Convert to Fahrenheit

        # Normalize and apply color map
        gray8_frame = np.zeros_like(gray16_frame, dtype=np.uint8)
        gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
        gray8_frame = np.uint8(gray8_frame)
        gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

        # Draw marker and temperature text
        cv2.circle(gray8_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
        cv2.putText(gray8_frame, "{0:.1f}°F".format(temperature_pointer),
                    (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # Save output image
        output_path = output_dir / f"thermal_output_{idx}.png"
        cv2.imwrite(str(output_path), gray8_frame)
        print(f"Saved: {output_path}")

        # Only process the first page, remove break if multiple pages are needed
        break  

print("Processing complete.")
