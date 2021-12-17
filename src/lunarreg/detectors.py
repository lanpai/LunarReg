import cv2 as cv

class SIFTDetector:
    def __init__(self):
        nFeatures = 0
        nOctaveLayers = 3
        contrastThreshold = 0.04
        edgeThreshold = 10
        sigma = 1.6
        self.sift = cv.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

    def detect(self, im):
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
