import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import os

def getMatches(desList, i, j):
  matches = bf.knnMatch(desList[i], desList[j], k=2)
  # Apply ratio test
  good = []
  for m in matches:
    if m[0].distance < 0.5*m[1].distance:
      good.append(m)
  matches = np.asarray(good)
  return len(matches), matches

if __name__ == "__main__":
  ## Read all Images in An Array
  print("Reading all Images from the Directory /1")
  directory = './1'
  imagelist = [f for f in os.listdir(directory) if f.endswith('.jpg')]
  N_Images = len(imagelist)
  ScaleFactor = 8
  imgListCV = []
  for image in imagelist:
    img = cv2.imread(directory+'/'+image)
    height,width,_ = img.shape
    img = cv2.resize(img,(int(width/ScaleFactor),int(height/ScaleFactor)))
    imgListCV.append(img)
    # cv2.imshow('Image',img)
    # cv2.waitKey(30)

  # find the keypoints and descriptors with SIFT
  print("Find all keypoints and descriptors with SIFT")
  sift = cv2.xfeatures2d.SIFT_create()
  kpList, desList = [],[]
  for image in imgListCV:
    kp, des = sift.detectAndCompute(image,None)
    kpList.append(kp)
    desList.append(des)

  ## Arranging all images
  print("Estimating Order of Images by Count Match Matrix.")
  bf = cv2.BFMatcher()
  matchMatrix = np.zeros((N_Images,N_Images))
  for i in range(0,N_Images):
    print("Matching with Image {0}".format(i))
    for j in range(0,N_Images):
      if i==j:
        continue
      matchCount, matches = getMatches(desList,i,j)
      print("Match Count {0} : {1}".format(j ,matchCount))
      matchMatrix[i][j] = matchCount
    
  print(matchMatrix)
      

# img_ = cv2.imread('right.jpeg')
# img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
# img = cv2.imread('left.jpeg')
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# print(kp1[0:5],kp1[0:5])

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
# good = []
# for m in matches:
#   if m[0].distance < 0.5*m[1].distance:
#     good.append(m)
#     matches = np.asarray(good)

# if len(matches[:,0]) >= 4:
#   src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#   dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#   H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
# #print H
# else:
#   raise AssertionError("Can't find enough keypoints.")

# dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
# dst[0:img.shape[0], 0:img.shape[1]] = img
# cv2.imwrite('output.jpg',dst)
# plt.imshow(dst)
# plt.show()

# if __name__ == "__main__":
#   ## Read all Images
#   imagelist = [f for f in os.listdir('./1') if f.endswith('.jpg')]
#   print(imagelist)
