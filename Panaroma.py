import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange




img_ = cv2.imread('right.jpeg')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('left.jpeg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
def get_keypoints(img1,img2):
  kp1,des1 = sift.detectAndCompute(img1,None)
  kp2,des2 = sift.detectAndCompute(img2,None)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)
  

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# print(kp1[0].__class__.__dict__)
# print(kp1[0].angle,kp1[0].octave, kp1[0].response,kp1[0].size,kp1[0].pt)
# print(des1[1])
# print(kp1,des1)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# print(matches[0][0].__class__.__dict__, matches[0][1].imgIdx)

# assert(False)

# print(len(matches))
# Apply ratio test
good = []
for m in matches:
  if m[0].distance < 0.5*m[1].distance:
    good.append(m)
matches = np.asarray(good)
    
# print(list(m.pt for m in matches))
# assert(False)


if len(matches[:,0]) >= 4:
  src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
  dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
  H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print H
else:
  raise AssertionError("Can't find enough keypoints.")
# dst = cv2.warpAffine(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]+img_.shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()

