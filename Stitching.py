import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import os
import math

indicexMax = lambda x: np.unravel_index(np.argmax(x),x.shape)
IntArray = lambda y: np.vectorize(lambda x: int(x))(y)
constructRT = lambda minX, minY : np.array([[1, 0, -minX],[0, 1, -minY],[0, 0, 1]])

def getMask(s):
  sGrey = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
  ret, KeypointMask = cv2.threshold(sGrey, 1, 255, cv2.THRESH_BINARY)
  return KeypointMask

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

def getAffinefromMatched(matches,kp1,kp2):
  if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    a, inliers = cv2.estimateAffinePartial2D(src,dst, cv2.RANSAC)
    a = np.concatenate((a,np.array([[0,0,1]])))
    # H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
  return a,None

def getFinalDimension(CoordinatesCombined):
  minX,minY = CoordinatesCombined.min(axis=0)
  maxX,maxY = CoordinatesCombined.max(axis=0)
  Width = maxX-minX
  Height = maxY-minY
  return minX,minY,Width,Height,Height*Width

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
    # minX,minY,Width,Height,Area = getFinalDimension(CoordinatesCombined)

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
  
def getFeatureMatchAndProjectionMatrix(desList,kpList,mode='h'):
  N = len(desList)
  matchMatrix = np.zeros((N,N))
  homographyMatrix = []
  for i in range(0,N):
    homographyArray = []
    for j in range(0,N):
      if i==j:
        homographyArray.append(np.identity(3))
        continue
      matchCount, matches = getMatches(desList[j],desList[i])
      if matchCount >= 4:
        if mode == 'a':
          H,_ = getAffinefromMatched(matches,kpList[j],kpList[i])
        else:
          H,_ = getHomographyFromMatched(matches,kpList[j],kpList[i])
        matchMatrix[i][j] = matchCount
        homographyArray.append(H)
      else:
        homographyArray.append(np.identity(3))
    homographyMatrix.append(homographyArray)
  matchMatrix = (matchMatrix+matchMatrix.T)/2
  return IntArray(matchMatrix) ,np.array(homographyMatrix)

def ContructFinalImage(CoordinatesCombined,HomographyArray,imgList):
  minX,minY,Width,Height,Area =  getFinalDimension(CoordinatesCombined)
  print("Constructing Image From Coordiantes & Homographies ......")
  print("Width: {0}, Height: {1}, Area: {2}".format(Width,Height,Area))
  RT = constructRT(minX,minY)
  HomographyArray = [RT.dot(H) for H in HomographyArray]
  N_Images = len(imgList)
  dst = np.zeros(shape=[Height, Width , 3], dtype=np.uint8)
  for k in range(N_Images):
    img = imgList[k]
    H = HomographyArray[k]
    s = cv2.warpPerspective(img,H, (Width,Height))
    mask = getMask(dst)
    s[np.where(mask==255)] = dst[np.where(mask==255)]
    dst = s
  return dst

def get2Dindices(matrix):
  try:
    r,c = indicexMax(matrix)
    return r,c
  except:
    r,c = 0,indicexMax(matrix)[0]
    return r,c


if __name__ == "__main__":
  ## ============ Settings ============ ##
  print("Enter the diretory of images..")
  d = input()
  print("Enter the Perspective Homography/Affine")
  mod = input()
  directory = './'+d
  imagelist = [f for f in os.listdir(directory) if f.endswith('.jpg')]
  N_Images = len(imagelist)
  ScaleFactor = 8

  ## ======== Initializing Lists and Functions ======= ##
  
  imgListCV, kpList, desList = [],[],[]
  sift = cv2.xfeatures2d.SIFT_create()
  bf = cv2.BFMatcher()

  ## ===================================================================================================================
  ## Read all Images in An Array
  ## find the keypoints and descriptors with SIFT and Add to List
  ## ===================================================================================================================
  
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

  ## ===================================================================================================================
  ## Compute Feature Martrix and MAtrix of Homographies/ Affines
  ## ===================================================================================================================
  featureMatchMatrix,homographyMatrix = getFeatureMatchAndProjectionMatrix(desList,kpList,mode=mod)
  featureMatchMatrixSum = featureMatchMatrix.sum(axis=0)
  maxCount = featureMatchMatrixSum.max() 

  ## ===================================================================================================================
  ## Loop Over Each Image as the Base
  ## ===================================================================================================================
  
  MinArea = math.inf
  HomographyArrayFinal = None
  CoordinatesCombinedFinal = None
  ImageListFinal = None
  HomographyListFinal = None
  DimensionListFinal = None

  ## ===================================================================================================================
  ## Homography And Affine Projection
  ## ===================================================================================================================
  for Root in range(0,N_Images):
    if featureMatchMatrixSum[Root] > 0.25*maxCount:
      print("\nTaken Image {0} as the Base Image".format(Root))
    else:
      print("\nImage {0} Rejected as Base Image".format(Root))
      continue

    ## ============ Lists / Sets Initialization ============== ##
    CoordinatesCombined = getImageCoordinates(imgListCV[Root],mode='2D')
    HomographyArray = homographyMatrix[Root]
    ImageSet = np.arange(0,N_Images).tolist()
    ImageTaken = []
    HomographyList = []
    DimensionList = []
    RTIntial = np.identity(3)
    Area = imgListCV[Root].shape[0]*imgListCV[Root].shape[0]

    ## ============ Picking Images from Set One-One ============ ##
    TransferImage(Root,ImageSet,ImageTaken)
    assert len(ImageSet)+len(ImageTaken)==N_Images
   
    while len(ImageSet) != 0:
      rowIdx = np.array(ImageTaken)
      colIdx = np.array(ImageSet)
      featureMatchMatrixRound = featureMatchMatrix[rowIdx[:,None],colIdx]
      # print(ImageTaken, ImageSet, rowIdx,colIdx,featureMatchMatrixRound)

      ## ========== Take One iamge from Image Set ========== ##
      r,c = get2Dindices(featureMatchMatrixRound)
      parentNode = rowIdx[r]
      NodeSelected = colIdx[c]
      print("Parent Image: {0}, Selected Image {1}".format(parentNode,NodeSelected))

      ## ========== Compute Homogrpahy And Save ========== ##
      Hparent = HomographyArray[parentNode]
      H = homographyMatrix[parentNode][NodeSelected]
      HomographyArray[NodeSelected] = Hparent.dot(H)
      assert HomographyArray[NodeSelected].shape == (3,3)
      # print(Hparent, H, HomographyArray[NodeSelected])

      ## ========== Estimate Coordinates ========== ##
      warpedCoordinates = getWarpedImageCoordinates(imgListCV[NodeSelected], HomographyArray[NodeSelected])
      CoordinatesCombined = np.concatenate((CoordinatesCombined,warpedCoordinates),axis=0)
      # print(CoordinatesCombined,warpedCoordinates)
      
      ## ========== Transfer Image to Another Set ========== ##
      TransferImage(NodeSelected,ImageSet,ImageTaken)
      assert len(ImageSet) + len(ImageTaken) == N_Images

      ## ========== Compute Sequencial Homography ========== ##
      minX,minY,Width,Height,AreaNew = getFinalDimension(CoordinatesCombined)
      RTNew = constructRT(minX,minY)
      RTInitalInv = np.linalg.inv(RTIntial)
      RTBase = RTNew.dot(RTInitalInv)
      RTIntial = RTNew
      HBase = RTBase.dot(HomographyArray[Root])
      HWarp = RTNew.dot(HomographyArray[NodeSelected])
      DimensionList.append((Width,Height))
      HomographyList.append((HBase,HWarp))
      assert Area <= AreaNew
      Area = AreaNew

    ## ========== Comaprison for Best Fit of Base Image ========== ##
    minX,minY,Width,Height,Area = getFinalDimension(CoordinatesCombined)
    dst = ContructFinalImage(CoordinatesCombined,HomographyArray,imgListCV)
    cv2.imwrite('outputStitching{0}.jpg'.format(Root),dst)
    if Area<MinArea:
      MinArea = Area
      HomographyListFinal = HomographyList
      DimensionListFinal = DimensionList
      CoordinatesCombinedFinal = CoordinatesCombined
      HomographyArrayFinal = HomographyArray
      ImageListFinal = np.array(imgListCV)[np.array(ImageTaken)]

  ## ===================================================================================================================
  ## COmpute Final Image with Blending
  ## ===================================================================================================================
  dst = ContructFinalImage(CoordinatesCombinedFinal,HomographyArrayFinal,imgListCV)
  cv2.imwrite('outputStitchingFinal.jpg',dst)
