import os
from PIL import Image
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


TEST_IMAGE = 1000

path = "MY_data/train/*/*.jpeg"

images_list = []
image_data_list = []
try:
    for file in glob.glob(path):
        im = Image.open(file)
        images_list.append(im)
        image_data_list.append(np.asarray(im))
        im.close()
    print(f"Imported: {len(images_list)} images")
except:
    print("Error when importing images.")

print(image_data_list[TEST_IMAGE].shape)
print(image_data_list[TEST_IMAGE][0][0].shape)
print(image_data_list[TEST_IMAGE][100][100])



print(images_list[0].getcolors())


# tuple to select colors of each channel line
colors = ("red", "green", "blue")

# create the histogram plot, with three lines, one for
# each color


plt.imshow(image_data_list[TEST_IMAGE])
plt.figure()
plt.xlim([0, 256])
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
        image_data_list[TEST_IMAGE][:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()
