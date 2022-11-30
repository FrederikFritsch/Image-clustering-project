import numpy as np
import os


def saveFeaturesInCSV(folder, filename, features):
    os.makedirs(f'{folder}{filename}', exist_ok=True)
    features.to_csv(f'{folder}{filename}/{filename}.csv')


def combine_images(columns, space, images):
    from PIL import Image
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


def get_image_paths(data_path):
    all_paths = []
    print("Getting paths")
    #print("Data Path is : " + str(data_path))
    #print(os.walk(data_path))
    for index, directories in enumerate(os.walk(data_path)):
        #print(directories)
        for sample in directories[2]:
            # print(sample)
            if sample.endswith('.png'):
                full_path = directories[0] + "/" + sample
                all_paths.append(full_path)
    print(len(all_paths))
    return all_paths


def create_gabor_filters(kernelsize=[10], thetarotations=2, sigmas=[3], lamdas=[2.*np.pi], gammas=[0.4]):
    import cv2 as cv
    kernels = []
    for ksize in kernelsize:
        for theta in range(thetarotations):        # Thetarotations
            theta = theta / float(thetarotations) * np.pi
            for sigma in sigmas:                   # SIGMA with 1 and 3
                for lamda in lamdas:               # range of wavelengths
                    for gamma in gammas:           # GAMMA values of 0.05 and 0.5
                        phi = 0
                        kernel = cv.getGaborKernel(
                            (ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv.CV_32F)
                        kernels.append(kernel)
    return kernels
