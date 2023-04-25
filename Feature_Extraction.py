#Created by Bangash, Software Engineer CAS SIAT China. 

import cv2
import numpy as np

# Load image
image_path = 'C:/Users/admin/folder/image.PNG'
img = cv2.imread(image_path, 0)

# Apply thresholding to extract features
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours of the features
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw contours on
contour_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Draw contours on the blank image
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', thresh)
cv2.imshow('Contour Image', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
