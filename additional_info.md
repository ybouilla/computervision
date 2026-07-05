# Computer vision basics

HOG + SVM detectors : for object detections
HOG: Histogram of Oriented Gradients


## feature detectors and/or descriptors
A feature descriptor is an algorithm which takes an image and outputs feature descriptors/feature vectors. Feature descriptors encode interesting information into a series of numbers and act as a sort of numerical “fingerprint” that can be used to differentiate one feature from another.
(methodes de description de caracteristiques)
### SIFT (Scale-Invariant Feature Transform)

### SURF (Speeded-Up Robust Features)
SURF approximates expensive computations using:

Integral images

Box filters instead of Gaussian filters

### ORB (Oriented FAST and Rotated BRIEF)

### Harris conner detection
https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

detects intersections or edges of a structure
how it detects: Corners occur where intensity changes in two directions.

### Hough transform (line and circle)

line -> detects line in an image
cercle -> detects cercles in an image
elitpic..

### border detection : 
otsu methods


## Segmentation

watershed algo
region growing

# image enhancment

improve sharpening  : laplacian filter
improve contrast: histogram equalization

# simple image transf

https://www.sfu.ca/~jtmulhol/py4math/linalg/ap-image-basics/ 

# 3D images

https://medium.com/red-buffer/beyond-the-surface-advanced-3d-mesh-generation-from-2d-images-in-python-0de6dd3944ac
https://lacie-life.github.io/posts/3D-DL-3/
https://medium.com/red-buffer/mastering-3d-spaces-a-comprehensive-guide-to-coordinate-system-conversions-in-opencv-colmap-ef7a1b32f2df

## cameras
https://pytorch3d.org/docs/cameras
https://pytorch3d.org/docs/renderer_getting_started


## Github repo

https://github.com/NVlabs/neuralangelo
https://github.com/dunbar12138/pix2pix3d

## opencv courses

https://www.geeksforgeeks.org/python/opencv-python-tutorial/
https://github.com/PacktPublishing/Hands-On-Image-Processing-with-Python/tree/master

## image metrics Evaluation

https://github.com/Jingnan-Jia/segmentation_metrics
 