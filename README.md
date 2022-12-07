# Software for Image Clustering
Authors: Frederik Fritsch, Lu Chen, Vladislav Bertilsson

## Requirements:
- Python >3.9.12
- Opencv 4.6.0.66
- Numpy 1.23.4
- Tensorflow 2.10.0
- Pandas 1.5.1


## Modules
### Traditional feature extraction
This module extracts features from images without using pre-trained neural networks. The following methods are currently available:
* Color distribution
* Color moments
* Local Binary Patterns
* ROI color distribution
* Contour moments
* ORB descriptors

### Neural Network feature extraction
This module extracts feature from images with pre-trained neural networks. The following models are available:
* VGG16
* Extractor
### K-Means clustering
This module performs clustering through K-Means. 
### DBSCAN clustering
This module performs clustering through DBSCAN.
### HDBSCAN clustering
This module performs clustering through HDBSCAN.
### Evaluation and plotting module
This module plots merged images for inspection evaluation and more.
