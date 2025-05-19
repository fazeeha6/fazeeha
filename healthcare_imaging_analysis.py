import cv2
import numpy as np

# Load MRI scan image (grayscale)
image = cv2.imread('brain_mri.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply thresholding to segment the tumor
_, thresholded = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

# Perform morphological operations to remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

# Find contours (likely tumor boundaries)
contours, _ = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

# Show results
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded)
cv2.imshow('Segmented Tumor', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
