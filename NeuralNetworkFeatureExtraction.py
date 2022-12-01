import sys
from src.Utils import *
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model
from src.featureExtraction import *
import time

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        data_path = str(args[0])
        filename = str(args[1])
        image_size = (int(args[2]), int(args[2]))
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()
all_image_paths = get_image_paths(data_path)
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
starttime = time.time() 

df = pd.DataFrame()
for path in all_image_paths:
    np_array_features = dnn_feature_exctration(path, model)
    feat = pd.DataFrame(np_array_features)
    feat.insert(0, "Name", path)
    df = pd.concat([df, feat], ignore_index=True)

endtime = time.time() 
print(f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
saveFeaturesInCSV("Results/DNN/", filename, df)
