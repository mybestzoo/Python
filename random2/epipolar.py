import cv2
import numpy as np
from matplotlib import pyplot as plt

# # Function to draw epilines
# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         cv2.line(img1, (x0,y0), (x1,y1), color,1)
#         cv2.circle(img1,tuple(pt1),5,color,-1)
#         cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

img1 = cv2.imread('May1L.jpg',0)  #queryimage # left image
img2 = cv2.imread('May1R.jpg',0) #trainimage # right image

sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

#Fundamental matrix
pts1 = np.float32(pts1)#pts1 = np.int32(pts1)
pts2 = np.float32(pts2)#pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
#
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
#
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()

# # Calculate rectification homographies
# H1 = np.zeros((3,3))
# H2 = np.zeros((3,3))
# cv2.stereoRectifyUncalibrated(pts1.reshape((pts1.shape[0] * 2, 1)),pts2.reshape((pts2.shape[0] * 2, 1)),F,img1.shape,H1,H2)
#
# # Rectify images
# img3 = cv2.warpPerspective(img1, H1, (960,1280))
# img4 = cv2.warpPerspective(img2, H2, (960,1280))
# # plt.subplot(121),plt.imshow(img3)
# # plt.subplot(122),plt.imshow(img4)
# # plt.show()

stereo = cv2.StereoSGBM(-128, 128, 15)
disp = stereo.compute(img2,img1)
plt.imshow(disp)
plt.show()