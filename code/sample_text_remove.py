import cv2
import numpy as np
import easyocr

# Load the image
image_path = r"D:\Project_detection\Updated_code\validated_code\sample_image_for_text_removing_1.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Detect text in the image
results = reader.readtext(image)

# Create a mask for inpainting
mask = np.zeros_like(gray)

for (bbox, text, prob) in results:
    if any(char.isdigit() for char in text) or any(ord(char) > 255 for char in text):  # Detect digits
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(bottom_right[0], top_right[0]))
        y_max = int(max(bottom_right[1], bottom_left[1]))

        # Draw filled rectangle on mask
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

# Apply inpainting to remove text
inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

# Save the output
output_path = r"D:\\Project_detection\Updated_code\\validated_code\\cleaned_image_final_3.jpg"
cv2.imwrite(output_path, inpainted_image)

print(f"Processed image saved at: {output_path}")
