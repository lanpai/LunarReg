import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

def colorTransfer(source, dest):
    source_gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    dest_gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)

    s_mean, s_std = cv.meanStdDev(source_gray, mask=cv.inRange(source_gray, 0, 254))
    d_mean, d_std = cv.meanStdDev(dest_gray, mask=cv.inRange(dest_gray, 0, 254))

    s_mean, s_std = np.hstack(s_mean)[0], np.hstack(s_std)[0]
    d_mean, d_std = np.hstack(d_mean)[0], np.hstack(d_std)[0]

    return cv.cvtColor(
            np.clip(((dest_gray-d_mean)*(s_std/d_std))+s_mean, 0, 255).astype(np.uint8),
            cv.COLOR_GRAY2BGR)

class IterativeFBM:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher
        self.maxIterations = 10
        self.hmgTolerance = 1.5

    def reprojTest(self, ptA, ptB, homography):
        ptA = np.array(ptA)
        ptB = np.array(ptB)

        dPt = ptA - ptB
        maxSqr = self.hmgTolerance*self.hmgTolerance
        distSqr = dPt[0]*dPt[0] + dPt[1]*dPt[1]

        return not dPt[0]*dPt[0] + dPt[1]*dPt[1] > maxSqr

    def match(self, imA, imB):
        # Color transfer
        imA = colorTransfer(imB, imA)

        # Match image heights
        height = min(imA.shape[0], imB.shape[0])
        imA = cv.resize(imA, dsize=(round((height/imA.shape[0])*imA.shape[1]), height))
        imB = cv.resize(imB, dsize=(round((height/imB.shape[0])*imB.shape[1]), height))

        # Initialize data
        kpB, desB = self.detector.detect(imB)

        homography = np.identity(3)
        chaoticMatches = []
        orderlyKeypoints = []
        orderlyDescriptors = []
        orderlyMatches = []

        # Iteration
        i = 1
        self.maxIterations = 50
        while i <= self.maxIterations:
            print(f'Iteration: {i}')

            # Chaotic step
            imAprime = cv.warpPerspective(imA, homography, imA.shape[:-1][::-1])
            kpAprime, desAprime = self.detector.detect(imAprime)
            matches = self.matcher.match(desAprime, desB)

            # Find homography
            ptsA, ptsB = [], []
            for match in matches:
                ptsA.append(kpAprime[match.queryIdx].pt)
                ptsB.append(kpB[match.trainIdx].pt)

            M, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, 3.)
            homography = M.dot(homography)
            print(M)
            print(homography)

            # Test points against homography
            matches = list(filter(lambda match:
                self.reprojTest(kpAprime[match.queryIdx].pt, kpB[match.trainIdx].pt, homography), matches))

            # TODO: Redundancy check

            for match in matches:
                orderlyMatches.append(cv.DMatch(
                    match.queryIdx + len(orderlyKeypoints), match.trainIdx, match.distance))

            # Debug plot (chaotic)
            #imMatch = cv.drawMatches(
            #        imAprime, kpAprime, imB, kpB, matches, None,
            #        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(imMatch)
            #plt.show()

            # Append chaotic data
            for kp in kpAprime:
                # Invert keypoints to original image space
                pt = [kp.pt[0], kp.pt[1], 1.]
                kp.pt = tuple(np.dot(np.linalg.inv(homography), pt)[:-1])
            orderlyKeypoints = orderlyKeypoints + list(kpAprime)
            orderlyDescriptors = orderlyDescriptors + list(desAprime)

            i = i + 1

        # Debug plot (orderly)
        imMatch = cv.drawMatches(
                imA, orderlyKeypoints, imB, kpB, orderlyMatches, None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(imMatch)
        plt.show()
