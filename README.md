# 3D-Image-Reconstruction-for-Ultrasound-images

Small Python examples for working with 2D ultrasound images: feature extraction by thresholding, and a 3D surface view of image intensities.

> Status: simple examples. Full 3D volume reconstruction from a series of ultrasound slices is **planned but not implemented yet**.

## Scripts
| Script | What it does |
|---|---|
| `Feature_Extraction.py` | Loads one grayscale image, applies Otsu thresholding, finds contours with OpenCV and shows the original, thresholded and contour images |
| `Kernel_Regression.py` | Loads one image, downsamples it, fits a Gaussian radial basis function (RBF) surface to the pixel intensities and plots it as a 3D surface with matplotlib |

The second script treats pixel intensity as a height value for visualisation. It is not a reconstruction from multiple slices.

## Requirements
```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

## How to run
1. Edit the image path at the top of each script (`image_path` or `path`) to point to your own image.
2. Run the script, for example `python Feature_Extraction.py`.

No images are included in this repository. Use your own images or a public dataset, and do not upload patient data.

## Planned
Stack a series of 2D slices into a 3D volume (for example with VTK `vtkImageData`) and render it.

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
