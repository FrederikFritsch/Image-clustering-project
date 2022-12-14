# Software for Image Clustering
Authors: Frederik Fritsch, Lu Chen, Vladislav Bertilsson

## Requirements:
- Python >3.9.12
- Opencv 4.6.0.66
- Numpy 1.23.4
- Tensorflow 2.10.0
- Pandas 1.5.1

Step-by-step guide:
1. Choose method for feature extraction (traditional/NN)
2. Choose clustering method (K-Means/DBSCAN/HDBSCAN)
3. Evaluate

We have provided scripts to run all modules on Alvis, but the python files can of be run on any computer.

## Modules
### Feature Extraction
#### Traditional feature extraction
This module extracts features from images without using pre-trained neural networks. The following methods are currently available:
* Color distribution
* Color moments
* Local Binary Patterns
* ROI color distribution
* Contour moments
* ORB descriptors

Example on how to run this file:
```python3 TraditionalFeatureExtraction.py Image_Data/ Test12 640 350```

#### Neural Network feature extraction
This module extracts feature from images with pre-trained neural networks. The following models are available:
* VGG16
* Extractor

Example on how to run this file:
```python3 ...```

### Clustering
#### K-Means clustering
This module performs clustering through K-Means.
Example on how to run this file:
```python3 KMeansClustering.py Test12/Test12.csv Test12 Normalize 0.8 10 20```

#### DBSCAN clustering
This module performs clustering through DBSCAN.
#### HDBSCAN clustering
This module performs clustering through HDBSCAN.

### Evaluation and plotting module
This module plots merged images for inspection evaluation and more.
Example on how to run this file:
```python3 Evaluation.py Test12/KMeansResults.csv Test12/ 15```
