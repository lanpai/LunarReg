import sys
sys.path.append('../src')

from os import listdir

import cv2 as cv
import matplotlib.pyplot as plt

from lunarreg import IterativeFBM
from lunarreg.detectors import SIFTDetector
from lunarreg.matchers import BFMatcher


# Load detectors and matchers
detector = SIFTDetector()
matcher = BFMatcher()
matcher.ratio = 0.65

# Load iterative FBM
fbm = IterativeFBM(detector, matcher)

def show_plot(imA, imB):
    print(f'{imA} / {imB}')

    imA = cv.imread(imA)
    imB = cv.imread(imB)

    try:
        (chaoticHomography, orderlyHomography,
                kpA, desA, kpB, desB,
                orderlyMatches, i, imMatch) = fbm.match(imA, imB)

        plt.imshow(imMatch)
        plt.show()
    except Exception as e:
        print(e)

show_plot('generated_loc_2_256x256.png', 'original_loc_2_256x256.png')
show_plot('generated_loc_7_256x512.png', 'original_loc_7_256x512.png')
