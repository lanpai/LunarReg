import cv2 as cv
import numpy as np

#import matplotlib.pyplot as plt

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
        self.maxIterations = 15
        self.reprojTolerance = 1.5
        self.redundancyTolerance = 3.

    def reprojTest(self, ptA, ptB, homography):
        ptA = cv.perspectiveTransform(
                np.array([[ptA]], dtype=np.float64),
                homography
                )[0,0]
        ptB = np.array(ptB)

        dPt = ptA - ptB
        maxSqr = self.reprojTolerance*self.reprojTolerance
        distSqr = dPt[0]*dPt[0] + dPt[1]*dPt[1]

        return distSqr <= maxSqr

    def match(self, imA, imB):
        # Color transfer
        imA = colorTransfer(imB, imA)

        # Match image heights
        height = min(imA.shape[0], imB.shape[0])
        imA = cv.resize(imA, dsize=(round((height/imA.shape[0])*imA.shape[1]), height))
        imB = cv.resize(imB, dsize=(round((height/imB.shape[0])*imB.shape[1]), height))

        # Initialize data
        kpB, desB = self.detector.detect(imB)

        chaoticHomography = np.identity(3)
        orderlyHomography = np.identity(3)
        orderlyKeypoints = []
        orderlyDescriptors = []
        orderlyMatches = []

        # Iteration
        i = 1
        while i <= self.maxIterations:
            # Chaotic step
            imAprime = cv.warpPerspective(imA, chaoticHomography, imA.shape[:-1][::-1])
            kpAprime, desAprime = self.detector.detect(imAprime)
            matches = self.matcher.match(desAprime, desB)

            # Find chaotic homography
            ptsA, ptsB = [], []
            for match in matches:
                ptsA.append(kpAprime[match.queryIdx].pt)
                ptsB.append(kpB[match.trainIdx].pt)
            try:
                M, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, 3.)
            except:
                try:
                    M, mask = cv.findHomography(np.array(ptsA), np.array(ptsB)) # Default to least-squares
                except Exception as e:
                    if i > 1: break
                    assert len(ptsA) >= 4, 'Less than 4 matches found for homography!'
                    raise e
            prevChaoticHomography = chaoticHomography.copy()
            chaoticHomography = M.dot(chaoticHomography)

            # Test points against chaotic homography reprojection
            matches = list(filter(lambda match:
                self.reprojTest(
                    kpAprime[match.queryIdx].pt, kpB[match.trainIdx].pt, M), matches))

            # Redundancy check
            for match1 in matches:
                shouldAppend = True
                pt1 = np.array(kpAprime[match1.queryIdx].pt)
                for j, match2 in enumerate(orderlyMatches):
                    # Don't check recently appended matches
                    if match2.queryIdx >= len(orderlyKeypoints):
                        break

                    pt2 = np.array(orderlyKeypoints[match2.queryIdx].pt)

                    dPt = pt1 - pt2
                    maxSqr = self.redundancyTolerance*self.redundancyTolerance
                    distSqr = dPt[0]*dPt[0] + dPt[1]*dPt[1]

                    if distSqr <= maxSqr:
                        if match1.distance < match2.distance:
                            orderlyMatches.pop(j)
                            shouldAppend = True
                        else:
                            shouldAppend = False
                        break

                if shouldAppend:
                    orderlyMatches.append(cv.DMatch(
                        match1.queryIdx + len(orderlyKeypoints), match1.trainIdx, match1.distance))

            # Append chaotic data
            for kp in kpAprime:
                # Invert keypoints to original image space
                kp.pt = tuple(
                        cv.perspectiveTransform(
                            np.array([[kp.pt]], dtype=np.float64),
                            np.linalg.inv(prevChaoticHomography)
                            )[0,0])
            orderlyKeypoints = orderlyKeypoints + list(kpAprime)
            orderlyDescriptors = orderlyDescriptors + list(desAprime)

            if len(orderlyMatches) > 0:
                # Find orderly homography
                ptsA, ptsB = [], []
                for match in orderlyMatches:
                    ptsA.append(orderlyKeypoints[match.queryIdx].pt)
                    ptsB.append(kpB[match.trainIdx].pt)
                try:
                    orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, 3.)
                except:
                    try:
                        orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB)) # Default to least-squares
                    except Exception as e:
                        if i > 1: break
                        assert len(ptsA) >= 4, 'Less than 4 matches found for homography!'
                        raise e

                # Test points against orderly homography reprojection
                matches = list(filter(lambda match:
                    self.reprojTest(
                        orderlyKeypoints[match.queryIdx].pt, kpB[match.trainIdx].pt, orderlyHomography),
                        orderlyMatches))

            i = i + 1

        assert len(orderlyMatches) > 0, 'No matches found during iterative FBM!'

        # Debug plot (orderly)
        imMatch = cv.drawMatches(
                imA, orderlyKeypoints, imB, kpB, orderlyMatches, None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return (chaoticHomography, orderlyHomography,
                orderlyKeypoints, orderlyDescriptors,
                kpB, desB,
                orderlyMatches, i, imMatch)
