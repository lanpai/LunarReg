import sys
sys.path.append('../src')

from os import listdir

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from lunarreg import IterativeFBM
from lunarreg.detectors import SIFTDetector
from lunarreg.matchers import BFMatcher

# Load detectors and matchers
detector = SIFTDetector()
matcher = BFMatcher()
matcher.ratio = 0.65

# Load iterative FBM
fbm = IterativeFBM(detector, matcher)

def show_plot(
        imA, imB,
        chaoticKeypoints, orderlyKeypoints, kpB,
        chaoticMatches, orderlyMatches,
        chaoticHomography, orderlyHomography):
    fig = plt.figure()

    axOA = fig.add_subplot(321)
    axOA.axis('off')
    axOB = fig.add_subplot(322)
    axOB.axis('off')
    axA = fig.add_subplot(323)
    axA.axis('off')
    axB = fig.add_subplot(324)
    axB.axis('off')
    axC = fig.add_subplot(325)
    axC.axis('off')
    axD = fig.add_subplot(326)
    axD.axis('off')

    imAprimeC = cv.warpPerspective(imA, chaoticHomography, imA.shape[:-1][::-1])
    imAprimeO = cv.warpPerspective(imA, orderlyHomography, imA.shape[:-1][::-1])

    axOA.imshow(imA)
    axOB.imshow(imB)
    axA.imshow(imAprimeC)
    axB.imshow(imB)
    axC.imshow(imAprimeO)
    axD.imshow(imB)

    for match in chaoticMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[chaoticKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    chaoticHomography
                    )[0,0])
        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axA, axesB=axB, color='red')
        axB.add_artist(con)
    for match in orderlyMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[orderlyKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    orderlyHomography
                    )[0,0])
        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axC, axesB=axD, color='red')
        axD.add_artist(con)

    plt.show()

def match_images(imA, imB):
    print(f'{imA} / {imB}')

    imA = cv.imread(imA)
    imB = cv.imread(imB)

    (chaoticHomography, orderlyHomography,
            chaoticKeypoints, orderlyKeypoints,
            chaoticDescriptors, orderlyDescriptors,
            kpB, desB,
            chaoticMatches, orderlyMatches,
            imA, imB, i) = fbm.match(imA, imB)

    print('Chaotic Homography:\n', chaoticHomography)
    print('Orderly Homography:\n', orderlyHomography)
    print('Orderly Matches: \t', len(orderlyMatches))

    show_plot(
            imA, imB,
            chaoticKeypoints, orderlyKeypoints, kpB,
            chaoticMatches, orderlyMatches,
            chaoticHomography, orderlyHomography)

fbm.maxIterations = 2
match_images('generated_loc_2_256x256.png', 'original_loc_2_256x256.png')
match_images('original_loc_2_256x256.png', 'generated_loc_2_256x256.png')
match_images('generated_loc_7_256x512.png', 'original_loc_7_256x512.png')
