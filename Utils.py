import numpy as np
import os
from PIL import Image
import cv2 as cv

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
    #print(width_max, height_max)
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

def get_image_paths(full_data_dir_path):
    all_paths = []
    for index, directories in enumerate(os.walk(full_data_dir_path)):
        for sample in directories[2]:
            if sample.endswith('.png'):
                full_path = directories[0] + "/" + sample
                all_paths.append(full_path)
    return all_paths