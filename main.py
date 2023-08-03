import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Read image
image_path = './IMAGE_NAME'
img = cv2.imread(image_path)

# Instantiate text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_results = reader.readtext(img)

threshold = 0.25
detected_text_list = []

# Extract text with score above the threshold
for t_, t in enumerate(text_results):
    bbox, text, score = t
    if score > threshold:
        detected_text_list.append(text)

# Save the detected text to a text file with a sans-serif font
output_text_file = "detected_text.txt"
with open(output_text_file, "w", encoding="utf-8") as file:
    file.write("Detected Text:\n")
    for text in detected_text_list:
        file.write(text + "\n")

# Show the image with bounding boxes (optional)
for t_, t in enumerate(text_results):
    bbox, text, score = t
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

cv2.imshow('Detected Text', img)
cv2.waitKey(0)
cv2.destroyAllWindows()