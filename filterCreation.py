import numpy as np
import cv2 as cv

def create_gabor_filters(kernelsize = [10], thetarotations = 2, sigmas = [3], lamdas = [2.*np.pi], gammas = [0.4]):
    kernels = []
    for ksize in kernelsize:  
        for theta in range(thetarotations):        # Thetarotations
            theta = theta / float(thetarotations) * np.pi
            for sigma in sigmas:                   # SIGMA with 1 and 3
                for lamda in lamdas:               # range of wavelengths
                    for gamma in gammas:           # GAMMA values of 0.05 and 0.5
                        phi = 0
                        kernel = cv.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv.CV_32F)
                        kernels.append(kernel)
    return kernels