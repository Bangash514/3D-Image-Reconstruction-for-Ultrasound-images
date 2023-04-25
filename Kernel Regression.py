#Created by; Bangash SIAT CAS, Software Engineer

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, transform


# Set the path to the folder containing the ultrasound images
path = "/ImageFolderPath"

# Load an example ultrasound image
filename = os.path.join(path, "horiz.PNG")
image = io.imread(filename, as_gray=True)
image = transform.rescale(image, 0.25, anti_aliasing=True)

# Extract the (x,y) coordinates of the image pixels
x, y = np.where(image)

# Extract the grayscale values of the image pixels
z = image[x, y]

# Create a meshgrid
xi, yi = np.meshgrid(np.arange(0, image.shape[1], 10), np.arange(0, image.shape[0], 10))

rbf = Rbf(x, y, z, function='gaussian')
zi = rbf(xi, yi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='jet')
ax.plot_surface(xi, yi, zi, cmap='jet', alpha=0.5)
plt.show()
