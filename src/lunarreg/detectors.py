import cv2 as cv

class SIFTDetector:
    def __init__(self):
        # Defining SIFT parameters
        nFeatures = 0
        nOctaveLayers = 4
        contrastThreshold = 0.04
        edgeThreshold = 10
        sigma = 1.6

        # Initializing SIFT detector
        self.sift = cv.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

    def detect(self, im):
        # Convert input image to gray scale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = self.sift.detectAndCompute(gray, None)

        return kp, des
