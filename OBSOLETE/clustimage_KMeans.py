from clustimage import Clustimage
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob


#-------- VARIABLES ----------

IMG_SIZE =512
HOG = True
PCA = False
TEST = False
path = "MY_data/predict/*.jpeg"
path2 = "MY_data/train/*/*.jpeg"
path3 = "/MY_data/predict/"
path4 = "/MY_data/train/"

current_dir = os.getcwd()
print(current_dir + path3)
images = []


#------- INITIALIZE ----------

if HOG and not PCA:
    cl = Clustimage(method='hog',ext =['jpeg'], params_hog={'orientations':4, 'pixels_per_cell':(8,8), 'cells_per_block':(1,1)})
if PCA and not HOG:
    cl = Clustimage(method='pca',ext =['jpeg'], params_pca={'n_components': 0.90})
if PCA and HOG:
    cl = Clustimage(method='pca-hog')


#---------- HOG ONLY ---------

if TEST:
    #-------- IMPORT DATA ----------
    try:
        images = [cl.imread(file, dim=(IMG_SIZE,IMG_SIZE), colorscale=0, flatten=True) for file in glob.glob(path)]
        print(f"Imported: {len(images)} images")
    except Exception as e:
        print(e)
    img_hog = cl.extract_hog(images[1])

    fig,axs=plt.subplots(1,2)
    axs[0].imshow(images[1].reshape(IMG_SIZE,IMG_SIZE))
    axs[0].axis('off')
    axs[0].set_title('Preprocessed image', fontsize=10)
    axs[1].imshow(img_hog.reshape(IMG_SIZE,IMG_SIZE), cmap='binary')
    axs[1].axis('off')
    axs[1].set_title('HOG', fontsize=10)
    plt.show()


#----------- USING BUILT IN RESULTS METHOD ---------
mega_path = []
current_path = current_dir + path4

for index, directories in enumerate(os.walk(current_path)):
    if index > 0:
        for sample in directories[2]:
            if sample.endswith('.jpeg'):
                full_path = directories[0] + "/" + sample
            mega_path.append(full_path)
#print(mega_path)
if not TEST:
    #-------- IMPORT DATA ----------
    #current_path = current_dir + path3

    results = cl.fit_transform(mega_path, min_clust=4, max_clust=12)
    #results = cl.cluster(evaluate='silhouette', cluster='dbscan', cluster_space='low')
    #results = cl.cluster(cluster='agglomerative', method='dbindex')

    cl.clusteval.plot()

    cl.clusteval.scatter(cl.results['xycoord'])

    cl.dendrogram()
    if HOG:
        # Plot unique image per cluster
        cl.plot_unique(img_mean=False, show_hog=True)

        # Scatterplot
        cl.scatter(dotsize=50, zoom=0.5, img_mean=False)

        # Plot images per cluster or all clusters
        cl.plot(labels=3, show_hog=True)
    
    if PCA:
        cl.pca.plot()

        # Plot unique image per cluster
        cl.plot_unique(img_mean=False)
    
        # Scatterplot
        cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
    
        # Plot images per cluster or all clusters
        cl.plot(labels=8)
    
    plt.show()






   #try:
    #    current_path = current_dir + path3
    #    images = cl.import_data(current_path, flatten=True)
    #    print(f"Imported: {len(images)} images")
    #except Exception as e:
    #    print("ERROR")
    #    print(e)

#cv.imshow(f"Cluster", images[1].reshape((128, 128, 1)))
#cv.imshow(f"Cluster", img_hog.reshape((128, 128, 1)))
#cv.waitKey(0)



#try:
#    images = [cl.imread(file, dim=(244,244), colorscale=3, flatten=False) for file in glob.glob(path)]
#    print(f"Imported: {len(images)} images")
#except Exception as e:
#    print(e)