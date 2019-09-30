import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import os
import math
from math import sqrt
from shapely.geometry import Polygon,LineString,Point

indicexMax = lambda x: np.unravel_index(np.argmax(x),x.shape)
IntArray = lambda y: np.vectorize(lambda x: int(x))(y)
constructRT = lambda minX, minY : np.array([[1, 0, -minX],[0, 1, -minY],[0, 0, 1]])

def get_int_lines(poly1,poly2):
  int_poly = poly1.intersection(poly2)

  lines1,lines2 = [],[]
  pt1 = (int_poly.exterior.coords[0])
  for point in int_poly.exterior.coords:
    line = LineString([pt1,point])
    if (poly1.contains(Point(pt1)) or poly1.contains(Point(point))):
        lines1.append(line)
    elif (poly2.contains(Point(pt1)) or poly2.contains(Point(point))):
        lines2.append(line)
    pt1  = Point(point)
  # print(lines1.coords,lines2.coords)
  return lines1,lines2

def find_dis_from_line(p,lines):
  # p = Point(point)
  dis = p.distance(lines[0])
  for line in lines[1:]:
    dis = min(dis,p.distance(line))
  return dis

def metric3(point,lines1,lines2):
  w1 = find_dis_from_line(point,lines2)
  w2 = find_dis_from_line(point,lines1)
  summ = w1+w2
  return (w1/summ,w2/summ)

# def find_metric(point,poly,dists):
#     p = Point(point)
#     poly_coords = poly.exterior.coords

#     # points = 
#     metrics = []
#     pt1 = Point(poly_coords[0])
#     dis1 = p.distance(pt1)
#     for point,dist in zip(poly_coords[1:],dists):
#         dis2 = p.distance(Point(point))
#         metrics.append(abs((dis2+dis1-dist)**(0.5)))
#         dis1 = dis2
#     return min(metrics)

# def find_metric2(point,poly,dists):
#     p = Point(point)
#     poly_coords = poly.exterior.coords

#     # points = 
#     metrics = []
#     pt1 = Point(poly_coords[0])
#     # dis1 = p.distance(pt1)
#     for point,dist in zip(poly_coords[1:],dists):
#         line1 = LineString([pt1,Point(point)])
#         dis2 = p.distance(line1)
#         metrics.append(dis2)
#         pt1 = Point(point)
#     return min(metrics)

# def calculate_dists(poly):
#     poly_coords = poly.exterior.coords
#     dists = []
#     pt1 = Point(poly_coords[0])
#     for point in poly_coords[1:]:
#         dists.append(pt1.distance(Point(point)))
#         pt1 = Point(point)
#     return dists

# def calculate_image_metric(w,h,poly):
#     dists = calculate_dists(poly)
#     image_metrics = np.ndarray([h,w],dtype=np.float32)
#     for i in range(w):
#         for j in range(h):
#             image_metrics[j][i] = find_metric2(Point(i,j),poly,dists)
#     return image_metrics

def get_mask(w,h,quad):
  mask = np.ndarray([h,w],dtype = np.bool)
  for i in range(w):
    for j in range(h):
      mask[j][i] = quad.contains(Point(i,j))
  return mask

def get_quad(width,height,H):
  four_points = [0,0,0,0]
  four_points[0] = np.array([0,0,1])
  four_points[1] = np.array([width-1,0,1])
  four_points[2] = np.array([width-1,height-1,1])
  four_points[3] = np.array([0,height-1,1])
  for i in range(4):
    x =  np.matmul(H,four_points[i])
    x = np.array([x[0]/x[2],x[1]/x[2]],dtype= np.float32)
    four_points[i] = x
  return Polygon(four_points)

def get_new_mask(mask1,tx,ty,h,w):
  mask1_new = np.ndarray((h,w),dtype=np.bool)
  # print(int(tx),int(ty))
  
  for i in range(h):
    for j in range(w):
      if (i>=ty and j >=tx and i - ty< mask1.shape[0] and j -tx< mask1.shape[1]):
        mask1_new[i][j] = mask1[i-ty][j-tx]
      else:
        mask1_new[i][j] = 0
  return mask1_new
def shift_poly(poly,tx,ty):
  points = poly.exterior.coords
  newpoints = []
  for point in points:
    newpoints.append((point[0]+tx,point[1]+ty))
  return Polygon(newpoints)

def blend(img1,mask1,metric1,img2,mask2,metric2):

  assert(img1.shape == img2.shape)
  plt.imshow(mask1)
  plt.show()
  plt.imshow(mask2)
  plt.show()
  dst = np.ndarray(img1.shape,dtype = np.uint8)
  for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
      if (mask1[i][j] and mask2[i][j]):
        for k in range(3):
          dst[i][j][k] = np.uint8(min((metric1[i][j]*np.float32(img1[i][j][k]) + metric2[i][j]*np.float32(img2[i][j][k]))/(metric1[i][j] + metric2[i][j]),255))
        # dst[i][j] = np.array((metric1[i][j]*img1[i][j] + metric2[i][j]*img2[i][j])/(metric1[i][j] + metric2[i][j]),
      elif (mask1[i][j]):
        dst[i][j] = img1[i][j].copy()
      elif (mask2[i][j]):
        dst[i][j] = img2[i][j].copy()
      else:
        dst[i][j] = np.array([0,0,0],dtype=np.uint8)
  print(dst.shape)
  return dst

def blend2(img1,mask1,poly1,img2,mask2,poly2):
    plt.imshow(mask1)
    plt.show()
    plt.imshow(mask2)
    plt.show()
    assert(img1.shape == img2.shape)
    lines1,lines2 = get_int_lines(poly1,poly2)
    dst = np.ndarray(img1.shape,dtype = np.uint8)
    for i in range(img1.shape[0]):
      for j in range(img1.shape[1]):
        if (mask1[i][j] and mask2[i][j]):
          w1,w2 = metric3(Point([j,i]),lines1,lines2)
          for k in range(3):
            dst[i][j][k] = np.uint8(w1*img1[i][j][k] + w2*img2[i][j][k])
          # dst[i][j] = np.array(w1*img1[i][j] + w2*img2[i][j],ndtype=np.uint8)
        elif (mask1[i][j]):
          dst[i][j] = img1[i][j].copy()
        elif (mask2[i][j]):
          dst[i][j] = img2[i][j].copy()
        else:
          dst[i][j] = np.array([0,0,0],dtype=np.uint8)
    print(dst.shape)
    return dst
def combine(img1,H1,img2,H2,w,h,mask1,poly):
    print(w,h)
    img_1 = cv2.warpPerspective(img1,H1,(w,h))
    plt.imshow(img_1)
    plt.show()
    
    img_2 = cv2.warpPerspective(img2,H2,(w,h))
    plt.imshow(img_2)
    plt.show()
    print(img_1.shape,img_2.shape)
    tx = int(H1[0][2])
    ty = int(H1[1][2])
    print(tx,ty)
    mask1_new = get_new_mask(mask1,tx,ty,h,w)
    quad = get_quad(img2.shape[1],img2.shape[0],H2)
    mask2 = get_mask(w,h,quad)
    mask_new = mask1_new + mask2
    poly1_new = shift_poly(poly,tx,ty)
    poly_new = poly1_new.union(quad)
    # img1_metric = calculate_image_metric(w,h,poly1_new)
    # img2_metric = calculate_image_metric(w,h,quad)
    dst = blend2(img_1,mask1_new,poly1_new,img_2,mask2,quad)
    return (dst,mask_new,poly_new)
  
def panaroma(imglist,Hpairs,wh):
    img1 = imglist[0]
    mask1 = np.ones([img1.shape[0],img1.shape[1]],dtype=np.bool)
    poly1 = Polygon([(0,0),(img1.shape[1],0),(img1.shape[1],img1.shape[0]),(0,img1.shape[0])])

    for i in range(len(Hpairs)):
      H1,H2 = Hpairs[i]
      w,h = wh[i]
      img2 = imglist[i+1]
      img1,mask1,poly1 = combine(img1,H1,img2,H2,w,h,mask1,poly1)
    print(img1.shape)
    return img1
  
# def getMask(s):
#   sGrey = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
#   ret, KeypointMask = cv2.threshold(sGrey, 1, 255, cv2.THRESH_BINARY)
#   return KeypointMask

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
      matchCount, matches = getMatches(desList[j],desList[i])
      if matchCount >= 4:
        H,_ = getHomographyFromMatched(matches,kpList[j],kpList[i])
        matchMatrix[i][j] = matchCount
        homographyArray.append(H)
      else:
        homographyArray.append(np.identity(3))
    homographyMatrix.append(homographyArray)
  matchMatrix = (matchMatrix+matchMatrix.T)/2
  return IntArray(matchMatrix) ,np.array(homographyMatrix)

# def ContructFinalImage(CoordinatesCombined,HomographyArray,imgList):
#   minX,minY,Width,Height,Area =  getFinalDimension(CoordinatesCombined)
#   print("Constructing Image From Coordiantes & Homographies ......")
#   print("Width: {0}, Height: {1}, Area: {2}".format(Width,Height,Area))
#   RT = constructRT(minX,minY)
#   HomographyArray = [RT.dot(H) for H in HomographyArray]
#   N_Images = len(imgList)
#   dst = np.zeros(shape=[Height, Width , 3], dtype=np.uint8)
#   for k in range(N_Images):
#     img = imgList[k]
#     H = HomographyArray[k]
#     s = cv2.warpPerspective(img,H, (Width,Height))
#     mask = getMask(dst)
#     s[np.where(mask==255)] = dst[np.where(mask==255)]
#     dst = s
#   return dst

def get2Dindices(matrix):
  try:
    r,c = indicexMax(matrix)
    return r,c
  except:
    r,c = 0,indicexMax(matrix)[0]
    return r,c

def ConstructAlphaBlending(CoordinatesCombined,HomographyArray,imgList):
  _,_,Width,Height,_ = getFinalDimension(CoordinatesCombined)
  mask = np.zeros(shape=[Height, Width], dtype=np.uint8)
  for k in range(N_Images):
    s = cv2.warpPerspective(imgList[k], HomographyArray[k], (Width,Height))
    mask += 5*getMask(s)
  cv2.imshow('fr',mask)
  cv2.waitKey(4000)

if __name__ == "__main__":
  ## Settings
  x = input()
  directory = './'+x
  imagelist = [f for f in os.listdir(directory) if f.endswith('.jpg')]
  # imagelist = ['2.jpg','3.jpg','4.jpg']
  N_Images = len(imagelist)
  ScaleFactor = 8

  ## Initializing Lists and Functions
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
  ## Compute Feature Martrix and MAtrix of Homographies
  ## ===================================================================================================================
  featureMatchMatrix,homographyMatrix = getFeatureMatchMatrix(desList,kpList)
  featureMatrixRootSort = featureMatchMatrix.sum(axis=0)
  # print(featureMatchMatrix,featureMatrixRootSort,homographyMatrix)

  ## ===================================================================================================================
  ## Loop Over Each Image as the Base
  ## ===================================================================================================================
  
  MinArea = math.inf
  HomographyArrayFinal = None
  CoordinatesCombinedFinal = None
  ImageListFinal = None
  HomographyListFinal = None
  DimensionListFinal = None

  for Root in range(0,N_Images):
    print("Taken Image {0} as the Base Image".format(Root))

    ## Consider All The Sets
    CoordinatesCombined = getImageCoordinates(imgListCV[Root],mode='2D')
    HomographyArray = homographyMatrix[Root]
    ImageSet = np.arange(0,N_Images).tolist()
    ImageTaken = []
    # print(HomographyArray, CoordinatesCombined)

    TransferImage(Root,ImageSet,ImageTaken)
    assert len(ImageSet)+len(ImageTaken)==N_Images

    RTIntial = np.identity(3)
    HomographyList = []
    DimensionList = []
    Area = imgListCV[Root].shape[0]*imgListCV[Root].shape[0]

    while len(ImageSet) != 0:
      rowIdx = np.array(ImageTaken)
      colIdx = np.array(ImageSet)
      featureMatchMatrixRound = featureMatchMatrix[rowIdx[:,None],colIdx]
      # print(ImageTaken, ImageSet, rowIdx,colIdx,featureMatchMatrixRound)

      ## Take One iamge from Image Set 
      r,c = get2Dindices(featureMatchMatrixRound)
      parentNode = rowIdx[r]
      NodeSelected = colIdx[c]
      print("Parent Image: {0}, Selected Image {1}".format(parentNode,NodeSelected))

      ## Compute Homogrpahy And Save
      Hparent = HomographyArray[parentNode]
      H = homographyMatrix[parentNode][NodeSelected]
      HomographyArray[NodeSelected] = Hparent.dot(H)
      assert HomographyArray[NodeSelected].shape == (3,3)
      # print(Hparent, H, HomographyArray[NodeSelected])

      ## Estimate Coordinates
      warpedCoordinates = getWarpedImageCoordinates(imgListCV[NodeSelected], HomographyArray[NodeSelected])
      CoordinatesCombined = np.concatenate((CoordinatesCombined,warpedCoordinates),axis=0)
      # print(CoordinatesCombined,warpedCoordinates)
      
      ## Transfer Image to Another Set
      TransferImage(NodeSelected,ImageSet,ImageTaken)
      assert len(ImageSet) + len(ImageTaken) == N_Images

      ## Compute Sequencial Homography
      minX,minY,Width,Height,AreaNew = getFinalDimension(CoordinatesCombined)
      RTNew = constructRT(minX,minY)
      RTInitalInv = np.linalg.inv(RTIntial)
      RTBase = RTNew.dot(RTInitalInv)
      RTIntial = RTNew
      HBase = RTBase.dot(HomographyArray[Root])
      HWarp = RTNew.dot(HomographyArray[NodeSelected])

      t = [Width,Height,HBase,HWarp]
      assert Area <= AreaNew
      Area = AreaNew
      DimensionList.append((Width,Height))
      HomographyList.append((HBase,HWarp))


    ## Estimation of Final Size
    minX,minY,Width,Height,Area = getFinalDimension(CoordinatesCombined)
    # dst = ContructFinalImage(CoordinatesCombined,HomographyArray,imgListCV)
    # cv2.imwrite('outputStitching{0}.jpg'.format(Root),dst)
    if Area<MinArea:
      MinArea = Area
      HomographyListFinal = HomographyList
      DimensionListFinal = DimensionList
      CoordinatesCombinedFinal = CoordinatesCombined
      HomographyArrayFinal = HomographyArray
      ImageListFinal = np.array(imgListCV)[np.array(ImageTaken)]

  dst = panaroma(ImageListFinal,HomographyListFinal,DimensionListFinal)
  plt.imshow(dst)
  plt.show()
  # dst = ContructFinalImage(CoordinatesCombinedFinal,HomographyArrayFinal,imgListCV)
  
  cv2.imwrite('outputStitchingFinal.jpg',dst)
  
  # ConstructAlphaBlending(CoordinatesCombinedFinal,HomographyArrayFinal,imgListCV)

  ## For Blending
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