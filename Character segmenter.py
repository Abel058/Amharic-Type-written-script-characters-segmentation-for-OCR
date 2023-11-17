import cv2
import numpy as np
import os

# Load image and convert to grayscale
image = cv2.imread("ll.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

# Find contours of the characters
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Sort contours from top to bottom based on y-coordinate
cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

# Set the desired line height (in pixels)
line_height = 5

# Loop over each line of text
line_cnts = []
line_y_coords = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h >= line_height: # Only consider contours that have at least the desired line height
        if len(line_cnts) == 0 or y > line_y_coords[-1] + h:
            # This contour is on a new line
            line_cnts.append([c])
            line_y_coords.append(y)
        else:
            # This contour is on the same line as the previous one
            line_cnts[-1].append(c)

# Sort contours from left to right within each line
for i in range(len(line_cnts)):
    line_cnts[i] = sorted(line_cnts[i], key=lambda c: cv2.boundingRect(c)[0])

# Get the directory of the script and join it with the desired filename
script_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(script_directory, "output")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop over the contours and save each character as a separate image file
for i, line in enumerate(line_cnts):
    for j, c in enumerate(line):
        x, y, w, h = cv2.boundingRect(c)
        roi = thresh[y:y+h, x:x+w]
        char_index = j # Use j as char_index
        if char_index > 0:
            prev_x, _, prev_w, _ = cv2.boundingRect(line[char_index - 1])
            space_width = x - (prev_x + prev_w)
            if space_width > 10:  # Set your desired minimum space width here.it is in pixel
                space_image = np.zeros((h, space_width), dtype=np.uint8)
                space_path = os.path.join(output_directory, f"space_{i}_{char_index}.png")
                cv2.imwrite(space_path, space_image)

        char_path = os.path.join(output_directory, f"char_{i}_{j}.png")
        cv2.imwrite(char_path, roi)
