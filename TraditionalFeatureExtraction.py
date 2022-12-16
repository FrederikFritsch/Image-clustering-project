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
        dirname = str(args[1])
        image_size_x = int(args[2])
        image_size_y = int(args[3])
        colorfeature = int(args[4])
        ROIColorfeature = int(args[5])
        edgefeature = int(args[6])
        LBPfeature = int(args[7])
        orbfeature = int(args[8])

    except Exception as e:
        print("Wrong usage of arguments.Correct usage is:")
        print("Arg 1: Entire file path do Image Data")
        print("Arg 2: Directory name of results folder (Will be created automatically if no directory exists)")
        print("Arg 3: Image size X value")
        print("Arg 4: Image size Y value")
        print("Arg 5: 0 or 1 if you want to use Colorfeatures")
        print("Arg 5: 0 or 1 if you want to use ROI Colorfeatures")
        print("Arg 5: 0 or 1 if you want to use Edge Features")        
        print("Arg 5: 0 or 1 if you want to use LBP Features")
        print("Arg 5: 0 or 1 if you want to use ORB Features")
        print(e)
        quit()

    take_time = False
    base_dir = os.getcwd()

    # Get image paths in data directory
    all_image_paths = get_image_paths(data_path)


    if take_time:
        starttime = time.time()  # Start time
    # Feature Extraction
    dataframe_list = []
    for path in all_image_paths:
        dataframe = traditional_feature_extraction(path, (image_size_x, image_size_y), colorfeature, ROIColorfeature, edgefeature, LBPfeature, orbfeature)
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list, ignore_index=True)
    if take_time:
        endtime = time.time()  # Stop time
    if take_time:
        print(
            f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")

    saveDataFrameAsCSV("Results/", dirname, df)
    print("Finished feature extraction. Results are stored in /Results/<Directory Name>/")
