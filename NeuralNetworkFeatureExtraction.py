import sys
from src.Utils import *
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import Model
from src.featureExtraction import *
import time


def preTrainedModelPick(kerasModel):
    if kerasModel == "VGG16":
        model = VGG16()
        return Model(inputs=model.inputs, outputs=model.layers[-2].output), 224
    if kerasModel == "XCEPTION":
        model = Xception()
        return Model(inputs=model.inputs, outputs=model.layers[-2].output), 299
    else:
        raise Exception(
            "Wrong model input. Check the READ.ME for currently supported pre-trained models")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        data_path = str(args[0])
        filename = str(args[1])
        model_name = str(args[2])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()

#Get image paths
all_image_paths = get_image_paths(data_path)

model, image_size = preTrainedModelPick(model_name)


starttime = time.time()

print(image_size)
df = pd.DataFrame()
for path in all_image_paths:
    np_array_features = dnn_feature_exctration(path, model, image_size)
    feat = pd.DataFrame(np_array_features)
    feat.insert(0, "Name", path)
    df = pd.concat([df, feat], ignore_index=True)

endtime = time.time()
print(
    f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
saveDataFrameAsCSV("Results/", filename, df)
