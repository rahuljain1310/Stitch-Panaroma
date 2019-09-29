import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import os

indicexMax = lambda x: np.unravel_index(np.argmax(x),x.shape)

def showTestImage(image):
  cv2.imshow('test',image)
  cv2.waitKey(60)

def getMatches(desListi, desListj):
  matches = bf.knnMatch(desListi, desListj, k=2)
  good = []
  for m in matches:
    if m[0].distance < 0.5*m[1].distance:
      good.append(m)
  matches = np.asarray(good)
  return len(matches), matches

def mergeImage(img1,img2,ds1,ds2,kp1,kp2):
  pass

if __name__ == "__main__":
  ## Settings
  directory = './1'
  imagelist = [f for f in os.listdir(directory) if f.endswith('.jpg')]
  N_Images = len(imagelist)
  ScaleFactor = 8

  ## Initializing Lists and Functions
  imgListCV, kpList, desList = [],[],[]
  sift = cv2.xfeatures2d.SIFT_create()
  bf = cv2.BFMatcher()

  ## Read all Images in An Array
  ## find the keypoints and descriptors with SIFT and Add to List
  print("Reading all Images from the Directory {0}".format(directory))

  for image in imagelist:  
    img = cv2.imread(directory+'/'+image)
    height,width,_ = img.shape
    img = cv2.resize(img,(int(width/ScaleFactor),int(height/ScaleFactor)))
    kp, des = sift.detectAndCompute(img,None)
    kpList.append(kp)
    desList.append(des)
    imgListCV.append(img)
    # cv2.imshow('Image',img)
    # cv2.waitKey(30)

  while len(imgListCV)>1:
    N = len(imgListCV)
    matchMatrix = np.zeros((N,N))
    for i in range(0,N):
      print("Matching with Image {0}".format(i))
      for j in range(0,N):
        if i==j:
          continue
        matchCount, matches = getMatches(desList[i],desList[j])
        print("Match Count {0} : {1}".format(j ,matchCount))
        matchMatrix[i][j] = matchCount
    print(matchMatrix)
    i,j = indicexMax(matchMatrix)
    if j<i:
      i,j = j,i
    assert i<j

    ## Popping Chosen Images 
    img1 = imgListCV.pop(i)
    showTestImage(img1)
    img2 = imgListCV.pop(j-i-1)
    showTestImage(img2)
    ds1 = desList.pop(i)
    ds2 = desList.pop(j-i-1)
    kp1 = kpList.pop(i)
    kp2 = kpList.pop(j-i-1)
    assert len(imgListCV) == len(kpList) == len(desList) == N-2

    imgMerged = mergeImage(img1,img2,ds1,ds2,kp1,kp2)
    kp, des = sift.detectAndCompute(imgMerged,None)
    imgListCV.append(img)
    kpList.append(kp)
    desList.append(des)
    assert len(imgListCV) == len(kpList) == len(desList) == N-1
    
        

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
