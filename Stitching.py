import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from math import sqrt
from functools import reduce
import os

def get_line(p1,p2):
  a = p2[1]-p1[1]
  b = p1[0]-p2[0]
  c = p2[0]*p1[1]-p1[0]*p2[1]
  dist = sqrt(a*a + b*b)

  return (a/dist,b/dist,c/dist)

def get_distance(pt,line):
  (line[0]*pt[0] + line[1]*pt[1] + line[2])

def get_min_distance(pt,lineAr):
  return min(get_distance(pt,line) for line in lineAr)

def distance_from_edge(img):
  

def getMatches(img1, img2):
  matches = bf.knnMatch(img1,img2, k=2)
  # Apply ratio test
  good = []
  for m in matches:
    if m[0].distance < 0.5*m[1].distance:
      good.append(m)
  matches = np.asarray(good)
  return len(matches), matches

# def get_keypoints(img1,img2):
# def exposure_balancing(sift,img1,img2):
#   kp1,des1 = sift.detectAndCompute(img1,None)
#   kp2,des2 = sift.detectAndCompute(img2,None)
#   n_matches, matches = getMatches(img1,img2)
#   src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#   dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
def is_inside_image(img,i,j):
  if (i<img.shape[0] and j<img.shape[1]):
    return img[i][j] > 0
  else:
    return False
def exposure_balancing(img1,img2):
  intensity1 = 0
  intensity2 = 0
  for i in img2.shape[0]:
    for j in img2.shape[1]:
      if (img1[i][j] is not 0):
        if (is_inside_image(img1,i,j)):
          intensity1 += img1[i][j]
          intensity2 += img2[i][j]
  mean_exposure = (intensity1 + intensity2)/2
  ratio1 = mean_exposure/intensity1
  ratio2 = mean_exposure/intensity2
  for img in {img1,img2}:
    for i in img.shape[0]:
      for j in img.shape[1]:
        img[i][j] *= ratio1
      


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
      matchCount, matches = getMatches(desList[i],desList[j])
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
