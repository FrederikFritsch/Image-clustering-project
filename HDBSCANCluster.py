print("hello this is Lu")

import numpy as np
import pandas as pd
import os
import time
import sys
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from src.clusteringAlgorithms import *
from src.featureExtraction import *
from src.Utils import *


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        filepath = str(args[0])
        resultspath = str(args[1])
        normalization_method = str(args[2])
        pca_variance = float(args[3])
        min_clusters = int(args[4])
        max_clusters = int(args[5])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()
    


    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(e)
        quit()
    print(df)

    # Here you need to add your code
    # E.g normalizing, clustering, pca, evaluation etc


    os.makedirs(f'{resultspath}', exist_ok=True)  
    #results_df.to_csv(f'{resultspath}/HDBSCANResults.csv') 