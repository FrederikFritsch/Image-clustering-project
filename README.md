[![MIT licensed][shield-license]](#)

[shield-license]: https://img.shields.io/badge/license-MIT-blue.svg


# Software for Image Clustering
Authors: Frederik Fritsch, Lu Chen, Vladislav Bertilsson

## ⚙️ Requirements:
- [Python](https://www.python.org/) >3.9.12
- [Opencv](https://opencv.org/) 4.6.0.66
- [Numpy](https://numpy.org/) 1.23.4
- [Tensorflow](https://www.tensorflow.org/) 2.10.0
- [Pandas](https://pandas.pydata.org/) 1.5.1


Step-by-step guide:
1. Choose method for feature extraction (Traditional/DNN)
2. Choose clustering method (K-Means/DBSCAN/HDBSCAN)
3. Evaluate

We have provided scripts to run all modules on [Alvis](https://www.c3se.chalmers.se/about/Alvis/), but the python files can of be run on any computer.

## 🚀 Modules
This repo is structured using modules that can be executed independently. Each module stores the results in the /Results folder after execution.  

- The results from the "Feature Extraction Modules" can be used in the "Clustering Modules".  
- The results from the "Clustering Modules" can be used on the "Evaluation Module".  

![Modules](https://iili.io/HoD1lLX.png)
### Feature Extraction
Generate a dataframe and save it as a .csv with the following options:
* Traditional Feature Extration methods
* Pretrained Deep Neural Network 

 ------
#### Traditional Feature Extraction
This module extracts features from images without using pre-trained neural networks. The following methods are currently available:
* Color distribution
* Color moments
* Local Binary Patterns
* ROI color distribution
* Contour moments
* ORB descriptors

**Command to Run The File**  
```python3 TraditionalFeatureExtraction.py $DATA_PATH $CSV_FOLDER_NAME $WIDTH $HEIGHT $RESIZE_METHOD $COLORFEATURES $ROICOLORFEATURES $EDGEFEATURES $LBPFEATURES $ORBFEATURES```  
  
Where the inputs are the following:
* DATA_PATH = Path where the images are located
* CSV_FOLDER_NAME = Name of the folder where the resulting dataframe will be stored
* WIDTH = Width to resize the image to
* HEIGHT = Height to resize the image to
* RESIZE_METHOD = Method to use for resizing. Lanczos || Area || Linear || Cubic
* COLORFEATURES = Use color features? 0 || 1
* ROICOLORFEATURES = Use ROI color features? 0 || 1
* EDGEFEATURES = Use canny edge features? 0 || 1
* LBPFEATURES = Use LBP features? 0 || 1
* ORBFEATURES = Use ORB keypoint features? 0 || 1


Example on how to run this file:  
```python3 TraditionalFeatureExtraction.py Image_Data/ Test12 640 350 Lanczos 1 0 0 1 0```  

This will extract features from images inside the folder Image_Data/ and put the results in the folder Test12. The images will be resized to 640x350 using Lanczos algorithm and Color features + LBP features will be extracted.

---

#### Neural Network Feature Extraction
This module extracts feature from images with pre-trained neural networks

The following models are available:
* [VGG16](https://keras.io/api/applications/vgg/)
* [Xception](https://keras.io/api/applications/xception/)

**Command to Run The File**  
```python3 NeuralNetworkFeatureExtraction.py $DATA_PATH $CSV_FOLDER_NAME $MODEL_TYPE ```  
  
Where the inputs are the following:
* DATA_PATH = Path where the images are located
* CSV_FOLDER_NAME = Name of the folder where the resulting dataframe will be stored
* MODEL_TYPE = VGG16 || XCEPTION

Example on how to run this file:
```python3 KMeansClustering.py /Image_Data/ DNNTest VGG16 ```

---
### Clustering
#### K-Means Clustering
This module performs clustering through K-Means.
Example on how to run this file:
```python3 KMeansClustering.py Test12/Test12.csv Test12 Normalize 0.8 10 20```
#### DBSCAN Clustering
This module performs clustering through DBSCAN.
#### HDBSCAN clustering
This module performs clustering through HDBSCAN.  

---

### Evaluation and Plotting Module
This module plots merged images for inspection evaluation and more.
Example on how to run this file:
```python3 Evaluation.py Test12/KMeansResults.csv Test12/ 15```  

---
## 📝License
Licensed under the [MIT](https://github.com/FrederikFritsch/Image-clustering-project/blob/main/LICENSE.md) license.  


