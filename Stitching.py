import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import os

indicexMax = lambda x: np.unravel_index(np.argmax(x),x.shape)
IntArray = lambda y: np.vectorize(lambda x: int(x))(y)


def showTestImage(image):
  cv2.imshow('test',image)
  cv2.waitKey(2500)

def TransferImage(i,set1,set2):
  set1.remove(i)
  set2.append(i)

def getMatches(desListi, desListj):
  matches = bf.knnMatch(desListi, desListj, k=2)
  good = []
  for m in matches:
    if m[0].distance < 0.5*m[1].distance:
      good.append(m)
    matches = np.asarray(good)
  return len(matches), matches

def points3Dto2D(Coordinates):
  cn = []
  for row in Coordinates:
    x = row[0]/row[2]
    y = row[1]/row[2]
    cn.append([x,y])
  cn = np.array(cn)
  cn = np.vectorize(lambda  x: int(x)) (cn)
  return np.array(cn)

def getImageCoordinates(img,mode='3D'):
  h, w, _ = img.shape
  Coordinates = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]])
  if mode == '2D':
    Coordinates = points3Dto2D(Coordinates)
  return Coordinates

def getWarpedImageCoordinates(img,H):
  imgCoordinates = getImageCoordinates(img)
  NewCoordinates = H.dot(imgCoordinates.T)
  return points3Dto2D(NewCoordinates.T)  

def getHomographyFromMatched(matches,kp1,kp2):
  if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
  return H,masked

def getFinalDimension(CoordinatesCombined):
  minX,minY = CoordinatesCombined.min(axis=0)
  maxX,maxY = CoordinatesCombined.max(axis=0)
  Width = maxX-minX
  Height = maxY-minY
  return minX,minY,Width,Height

def mergeImage(img1,img2,ds1,ds2,kp1,kp2):
  _,matches = getMatches(ds1,ds2)
  showTestImage(img1)
  showTestImage(img2)
  print(matches.shape)
  if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    ## Get Estimated Coordinates in Warped Image
    CoordinatesImg1 = getImageCoordinates(img1,mode='2D')
    CoordinatesImg2 = getWarpedImageCoordinates(img2,H)
    CoordinatesCombined = np.concatenate((CoordinatesImg1,CoordinatesImg2),axis=0)
    # print(CoordinatesImg1,CoordinatesImg2,CoordinatesCombined)
    minX,minY = CoordinatesCombined.min(axis=0)
    maxX,maxY = CoordinatesCombined.max(axis=0)
    # print(minX,minY,maxX,maxY)
    Width = maxX-minX
    Height = maxY-minY
    # print(Width,Height)
    # minX,minY,Width,Height = getFinalDimension(CoordinatesCombined)

    RT = np.array([
      [1, 0, -minX],
      [0, 1, -minY],
      [0, 0, 1]
    ])
    NewH = RT.dot(H)

    dst = cv2.warpPerspective(img1, NewH, (Width,Height) )
    showTestImage(dst)
    dst[-minY:img2.shape[0]-minY, -minX:-minX+img2.shape[1]] = img2
    showTestImage(dst)

    cv2.imwrite('outputStitching.jpg',dst)
    return dst
  else:
    print("Cannot Find enough Keypoints.")
    return None
  
def getFeatureMatchMatrix(desList,kpList):
  N = len(desList)
  matchMatrix = np.zeros((N,N))
  homographyMatrix = []
  for i in range(0,N):
    homographyArray = []
    for j in range(0,N):
      if i==j:
        homographyArray.append(np.identity(3))
        continue
      matchCount, matches = getMatches(desList[i],desList[j])
      if matchCount >= 4:
        H,_ = getHomographyFromMatched(matches,kpList[i],kpList[j])
        matchMatrix[i][j] = matchCount
        homographyArray.append(H)
      else:
        homographyArray.append(np.identity(3))
    homographyMatrix.append(homographyArray)
  matchMatrix = (matchMatrix+matchMatrix.T)/2
  return IntArray(matchMatrix) ,np.array(homographyMatrix)

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
    print("Reading Image "+image) 
    img = cv2.imread(directory+'/'+image)
    height,width,_ = img.shape
    img = cv2.resize(img,(int(width/ScaleFactor),int(height/ScaleFactor)))
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(imgGrey,None)
    kpList.append(kp)
    desList.append(des)
    imgListCV.append(img)
    # cv2.imshow('Image',img)
    # cv2.waitKey(30)

  assert len(imgListCV) == len(kpList) == len(desList) == N_Images

  featureMatchMatrix,homographyMatrix = getFeatureMatchMatrix(desList,kpList)
  featureMatrixRootSort = featureMatchMatrix.sum(axis=0)
  print(featureMatchMatrix,featureMatrixRootSort)
  # print(homographyMatrix)

  for i in range(N_Images):
    Root = i
    CoordinatesCombined = getImageCoordinates(imgListCV(Root))
    HomographyArray = homographyMatrix[i]
    
    ImageSet = np.arange(0,N_Images).tolist()
    ImageTaken = []

    TransferImage(i,ImageSet,ImageTaken)
    assert len(ImageSet)+len(ImageTaken)==N_Images
    
    while len(ImageSet) != 0:
      rowIdx = np.array(ImageTaken)
      colIdx = np.array(ImageSet)
      featureMatchMatrixRound = featureMatchMatrix[rowIdx,colIdx]
      

      ## Take One iamge from Image Set 
      ## Compute Homogrpahy And Save
      ## Estimate size
      ## append coordinates
    

    


  ## Construct a Tree Using featureMatrix
  
  # while len(imgListCV)>1:
  #   N = len(imgListCV)
  #   matchMatrix = np.zeros((N,N))
  #   i,j = -1,-1
  #   if N == N_Images:
  #     print("To Select The Base Image")
  #     for i in range(0,N):
  #       print("Matching with Image {0}".format(i))
  #       for j in range(0,N):
  #         if i==j:
  #           continue
  #         matchCount, matches = getMatches(desList[i],desList[j])
  #         matchMatrix[i][j] = matchCount
  #     i,j = indicexMax(matchMatrix)
  #     if j<i:
  #       i,j = j,i
  #   else:
  #     print("To Add Images on the base Image")
  #     for j in range(0,N-1):
  #       matchCount, matches = getMatches(desList[i],desList[j])
  #       matchMatrix[N-1][j] = matchCount
  #     i = matchMatrix[N-1].argmax()
  #     j = N-1

  #   ## Compute Images to Match from matchMatrix
  #   print(matchMatrix,i,j)
  #   assert i<j

  #   ## Popping Chosen Images
  #   # showTestImage(imgListCV[i]) 
  #   # showTestImage(imgListCV[j])
  #   img1 = imgListCV.pop(i)
  #   img2 = imgListCV.pop(j-1)
  #   ds1 = desList.pop(i)
  #   ds2 = desList.pop(j-1)
  #   kp1 = kpList.pop(i)
  #   kp2 = kpList.pop(j-1)
  #   assert len(imgListCV) == len(kpList) == len(desList) == N-2

  #   imgMerged = mergeImage(img1,img2,ds1,ds2,kp1,kp2)
  #   imgGrey = cv2.cvtColor(imgMerged, cv2.COLOR_BGR2GRAY)
  #   ret, KeypointMask = cv2.threshold(imgGrey, 1, 255, cv2.THRESH_BINARY)
  #   # showTestImage(KeypointMask)

  #   kp, des = sift.detectAndCompute(imgGrey,mask=KeypointMask)
  #   imgListCV.append(img)
  #   kpList.append(kp)
  #   desList.append(des)

  #   assert len(imgListCV) == len(kpList) == len(desList) == N-1
  #   print("New Image Added at Position "+str(N-2))
    
        