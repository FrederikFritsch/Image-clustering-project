import numpy as np
import pandas as pd
import os
import time
import sys
from src.featureExtraction import traditional_feature_extraction
from src.Utils import *

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        data_path = str(args[0])
        filename = str(args[1])
        image_size_x = int(args[2])
        image_size_y = int(args[3])

    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()

    take_time = False
    base_dir = os.getcwd()

    # Get image paths in data directory
    all_image_paths = get_image_paths(data_path)

    gabor_filters = create_gabor_filters()  # Creates list of Gabor filters

    if take_time:
        starttime = time.time()  # Start time
    # Feature Extraction
    dataframe_list = []
    for path in all_image_paths:
        dataframe = traditional_feature_extraction(
            path, gabor_filters, (image_size_x, image_size_y))
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list, ignore_index=True)
    if take_time:
        endtime = time.time()  # Stop time
    if take_time:
        print(
            f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")

    saveDataFrameAsCSV("Results/", filename, df)
