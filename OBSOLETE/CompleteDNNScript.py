#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
from datetime import datetime


# In[2]:


base_dir= os.getcwd()
# change the working directory to the path where the images are located
tsne_df = pd.DataFrame()
# this list holds all the image filename
image_paths = []
data_dir = base_dir + "/Image_Data/ZAsaa1YOBks/"
# creates a ScandirIterator aliased as files
for index, directories in enumerate(os.walk(data_dir)):
    for sample in directories[2]:
        if sample.endswith('.png'):
            image_paths.append(data_dir + sample)
tsne_df["Image Name"] = image_paths
print(tsne_df)


# In[21]:


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


# In[4]:


def combine_images(columns, space, images):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    widths = []
    heights = []
    for image in images:
        img = Image.open(image)
        widths.append(img.width)
        heights.append(img.height)
        img.close()
    width_max = max(widths)
    height_max = max(heights)
    print(width_max, height_max)
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
        img.close()
    return background


# In[5]:


def perform_KMeans(data, min_clusters, max_clusters):
    # --------- CALCULATE K-MEANS CLUSTERS ------------
    sse = []
    silhouette_coefficients = []
    labels = []
    for nr_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(init = "random", n_clusters = nr_clusters, n_init = 10, max_iter=300, random_state = 22)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)
        labels.append(kmeans.labels_)
    return sse, score, silhouette_coefficients, labels


# In[6]:


# load model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


# In[7]:


data = {}

# lop through each image in the dataset
for flower in image_paths:
    feat = extract_features(flower, model)
    data[flower] = feat

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
feat.shape
(210, 1, 4096)

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)
feat.shape
(210, 4096)



# In[8]:


print(feat)


# In[10]:


# Standardize z
scaler = StandardScaler()
scaler.fit(feat)
feat = scaler.transform(feat)
print(feat)


# In[11]:


pca = PCA(0.95, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

print(x.shape)
print(x)


# In[12]:


# Kmeans K

min_cluster = 5
max_cluster = 20
sse, score, silhouette_coefficients, labels = perform_KMeans(x, min_cluster, max_cluster)


# In[13]:


kl = KneeLocator(range(min_cluster, max_cluster+1), sse, curve="convex", direction="decreasing")
print(kl.elbow)
n_clusters = np.argmax(silhouette_coefficients)+min_cluster
print(f"Silhouette coefficient: {n_clusters} clusters return best results")


# In[24]:


# Create Visual Evaluation
X = TSNE(n_components=2, perplexity=4).fit_transform(x)
cluster_labels = pd.Series(labels[np.argmax(silhouette_coefficients)])
tsne_df['ClusterID'] = cluster_labels.values
print(cluster_labels.values)
tsne_df["X_tsne"]  = X[:, 0]
tsne_df["Y_tsne"] = X[:, 1]


# In[25]:


#Plotting Elbow Method & Silhouette 

fig, axes = plt.subplots(2, 1)
plt.style.use("fivethirtyeight")
axes[0].plot(range(min_cluster, max_cluster+1), sse)
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("SSE")
axes[1].plot(range(min_cluster, max_cluster+1), silhouette_coefficients)
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Coefficient")

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="X_tsne", y="Y_tsne",
    hue="ClusterID",
    palette=sns.color_palette("hls", 10),
    data=tsne_df,
    legend="full",
    alpha=0.9
)
plt.show()


# In[26]:


date = datetime.now()
dir_name = date.strftime("%c")#.replace(":", "")
os.mkdir(base_dir+"/Results/DNN/"+dir_name)
results_path = base_dir+"/Results/DNN/"+dir_name+"/"
os.chdir(results_path)

for cluster_number in range(n_clusters):
    cluster = tsne_df.loc[tsne_df["ClusterID"]==cluster_number]
    image_list = []
    for image_path in cluster["Image Name"]:
        image_list.append(image_path)

    column_number = int(np.ceil(np.sqrt(len(image_list))))
    if len(image_list):
        merged_image = combine_images(columns=column_number, space=10, images=image_list)
        merged_image.save(str(cluster_number)+".png")
os.chdir(base_dir)


# In[ ]:




