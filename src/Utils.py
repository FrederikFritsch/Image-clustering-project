import numpy as np
import os


def saveDataFrameAsCSV(folder, filename, features):
    os.makedirs(f'{folder}{filename}', exist_ok=True)
    features.to_csv(f'{folder}{filename}/{filename}.csv')

def resize_image(image, size, method="Lanczos"):
    import cv2 as cv
    if method == "Lanczos":
        image = cv.resize(image, size, interpolation=cv.INTER_LANCZOS4)
    elif method == "Area":
        image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    elif method == "Linear":        
        image = cv.resize(image, size, interpolation=cv.INTER_LINEAR)
    elif method == "Cubic":
        image = cv.resize(image, size, interpolation=cv.INTER_CUBIC)
    else:
        image = cv.resize(image, size, interpolation=cv.INTER_NEAREST)
    return image
3
def combine_images(columns, space, images, distances, metric_string):
    from PIL import Image, ImageDraw
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
    #print(width_max, height_max)
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new(
        'RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        text = str(metric_string) + str(distances[i])
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        i1 = ImageDraw.Draw(img).text((10,5), text, (0,0,0))
        i2 = ImageDraw.Draw(img).text((10,15), text, (255,255,255))
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
        img.close()
    return background


def get_image_paths(data_path):
    all_paths = []
    print("Getting paths")
    #print("Data Path is : " + str(data_path))
    # print(os.walk(data_path))
    
    for index, directories in enumerate(os.walk(data_path)):
        # print(directories)
        for sample in directories[2]:
            # print(sample)
            if sample.endswith('.png'):
                full_path = directories[0] + "/" + sample
                all_paths.append(full_path)
    print(len(all_paths))
    return all_paths
