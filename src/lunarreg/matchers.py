import cv2 as cv
import numpy as np

def ratioTest(knnMatches, ratio):
    return list(filter(lambda match: match[0].distance <= match[1].distance*ratio, knnMatches))

def symmetryTest(knnMatchesA, knnMatchesB):
    matches = []
    for matchA in knnMatchesA:
        for matchB in knnMatchesB:
            if matchA[0].queryIdx == matchB[0].trainIdx and matchA[0].trainIdx == matchB[0].queryIdx:
                matches.append(cv.DMatch(matchA[0].queryIdx, matchA[0].trainIdx, matchA[0].distance))
    return matches

class BFMatcher:
    def __init__(self):
        normType = cv.NORM_L2
        crossCheck = False
        self.bf = cv.BFMatcher(normType, crossCheck=crossCheck)
        self.ratio = 0.65

    def match(self, desA, desB):
        # Ratio test using 2 nearest neighbors
        knnMatchesA = ratioTest(self.bf.knnMatch(desA, desB, 2), self.ratio)
        knnMatchesB = ratioTest(self.bf.knnMatch(desB, desA, 2), self.ratio)

        # Symmetry test
        matches = symmetryTest(knnMatchesA, knnMatchesB)

        return matches
