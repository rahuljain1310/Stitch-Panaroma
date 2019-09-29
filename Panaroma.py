import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from random import randrange
import os
import math
def is_inside_image(img,i,j):
  if (i<img.shape[0] and j<img.shape[1]):
    # return img[i][j][0] > 0
   
    return (img[i][j][0] > 0 or img[i][j][1] > 0 or img[i][j][2] > 0)
  else:
    return False
def exposure_balancing(img1,img2):
  intensity1 = [0,0,0]
  intensity2 = [0,0,0]
  int1 = int2 = 0
  mean_exposure =  [0,0,0]
  ratio= [[0,0,0],[0,0,0]]

  imghsv = [0,0]
  imghsv[0] = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
  imghsv[1] = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
  for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
      if (img2[i][j][0] > 0 or img2[i][j][1] > 0 or img2[i][j][2] > 0):
        if (is_inside_image(img1,i,j)):
          
          int1  = imghsv[0][i][j][2]
          # print(int1)
          int2  = imghsv[1][i][j][2]
          # print(int1,int2)
          meanint = np.uint8((np.uint16(int1)+np.uint16(int2))//2)
          imghsv[0][i][j][2] = min(255,meanint)
          imghsv[1][i][j][2] = meanint
          # print(int2)

  #       for k in range(3):
  #         if (is_inside_image(img1,i,j)):
            
  #           intensity1[k] += img1[i][j][k]
  #           intensity2[k] += img2[i][j][k]
  # print(intensity1[0],intensity1[1])
  # print(intensity2[0],intensity2[1])

  # mean_exposure = math.sqrt(int1*int2)
  # rt = [0,0]
  # rt[0] = mean_exposure/int1
  # rt[1] = mean_exposure/int2
  # for img,index in zip([img1,img2],[0,1]):
  #   for i in range(img.shape[0]):
  #     for j in range(img.shape[1]):
  #       imghsv[index][i][j][2] = min(int(rt[index]*imghsv[index][i][j][2]),255)
  
  img1 = cv2.cvtColor(imghsv[0],cv2.COLOR_HSV2BGR)
  img2 = cv2.cvtColor(imghsv[1],cv2.COLOR_HSV2BGR)
  return img1,img2
      
  # for k in range(3):
  #   mean_exposure[k] = math.sqrt(intensity1[k]*intensity2[k])
  #   # ratio[0][k] = mean_exposure[k]/intensity1[k]
    
  #   ratio[0][k] = 1
  #   ratio[1][k] = intensity1[k]/intensity2[k]
  #   # ratio[1][k] = mean_exposure[k]/intensity2[k]

  # print(ratio[0][0],ratio[1][0],ratio[0][1],ratio[1][1],ratio[0][2],ratio[1][2])
  # for img,index in zip([img1,img2],[0,1]):
  #   for i in range(img.shape[0]):
  #     for j in range(img.shape[1]):
  #       for k in range(3):
  #         img[i][j][k] = min(int(img[i][j][k]*ratio[index][k]),255)
  print(2)
def get_line(p1,p2):
  a = p2[1]-p1[1]
  b = p1[0]-p2[0]
  c = p2[0]*p1[1]-p1[0]*p2[1]
  dist = sqrt(a*a + b*b)

  return (a/dist,b/dist,c/dist)

def get_distance(pt,line):
  return abs(line[0]*pt[0] + line[1]*pt[1] + line[2])

def get_min_distance(pt,lineAr):
  return min(get_distance(pt,line) for line in lineAr)

def distance_from_edge(img,corners,mask):
  # distance_matrix = np.ndarray([img.shape[0],img.shape[1]],dtype = np.float)
  # def get_dist(i,j):
  #   if mask[]
  # for i in range(img.shape[0]):
  #   for j in range(img.shape[1]):
  #     if mask[i][j]:
  #       distance_matrix[i][j] = get_min_distance([i,j],corners[1])
  #     else:
  #       distance_matrix[i][j] = 0
  get_dist = lambda t: get_min_distance(t,corners[1])
  distance_matrix = np.vectorize(get_dist)(img)
  return distance_matrix

def weighted_add(dist1,val1,dist2,val2):
  return np.uint8((dist2*val1 + dist1*val2)/(dist1+dist2))

def blending(img1,img2,dist_mat1,dist_mat2,mask1,mask2):
  rows = img1.shape[0]
  cols = img1.shape[1]
  assert (rows == img2.shape[0] and cols == img2.shape[1])
  dst_img = np.ndarray([rows,cols,3],ndtype = np.uint8)
  dst_dist_mat = np.ndarray([img.shape[0],img.shape[1]],dtype = np.float)

  for i in range(rows):
    for j in range(cols):
      if (mask1[i][j] and mask2[i][j]):
        for k in range(3):
          dst_img[i][j][k] = weighted_add(dist_mat1[i][j],img1[i][j][k],dist_mat2[i][j],img2[i][j][k])
      elif (mask1[i][j]):
        dst_img[i][j] = np.array(img1[i][j]) 
      elif (mask2[i][j]):
        dst_img[i][j] = np.array(img2[i][j])
      else:
        dst_img[i][j] = np.array([0,0,0],dtype=np.uint8)
      dst_dist_mat[i][j] = min(dist_mat1[i][j],dist_mat)
  dest_mask = mask1+mask2

  return (dst_img,dest_mask)

def get_mask(img,corners):
  mask = np.ndarray([img.shape[0],img.shape[1]],dtype = np.bool)
  p =  path.Path(corners[0])
  is_inside = lambda t: p.contain_points(t)
  for i in img.shape[0]:
    mask[i] = p.contain_points(img[i])
  return mask
  # if (p.contain_points(img))
# def blending(img1,img2):
def blend_2image(img1,img2,mask1,distmat1,H,w,h):
  corners = get_lines(w,h,H)
  mask2 = get_mask(img2,corners)
  distmat2 = distance_from_edge(img2,corners,mask2)
  dest_img,dest_mask,dest_distmat = blending(img1,img2,distmat1,distmat2,mask1,mask2)

def get_lines(width ,height, H):
  four_points = np.ndarray(1,3)
  four_points[0] = np.array([0,0,1])
  four_points[1] = np.array([0,width-1,1])
  four_points[2] = np.array([height-1,0,1])
  four_points[3] = np.array([height-1,width-1,1])
  pt1 = None
  lines = []
  for i in range(4):
    x =  np.matmul(H,four_points[i])
    x = [x[0]/x[2],x[1]/x[2]]
    if pt1 is not None:
      lines.append(get_line(pt1,x)) 
    pt1 = x
    four_points[i] = x
  x = four_points[0]
  lines.append(get_line(pt1,x))
  return (four_points,lines)
   

img_ = cv2.imread('right.jpeg')
ScaleFactor = 2
height,width,_ = img_.shape
img_ = cv2.resize(img_,(int(width/ScaleFactor),int(height/ScaleFactor)))

img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
img = cv2.imread('left.jpeg')
height,width,_ = img.shape
img = cv2.resize(img,(int(width/ScaleFactor),int(height/ScaleFactor)))

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
# qr1 = [],qr2 = [],tr1 = [], tr2= []
# for m in matches 
# for kp in kp1:
#   qr1.append

if len(matches[:,0]) >= 4:
  src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
  dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
  H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print H
else:
  raise AssertionError("Can't find enough keypoints.")
# dst = cv2.warpAffine(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]+img_.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()

img,dst = exposure_balancing(img,dst)

dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()

if __name__ == "__main__":
  ## Read all Images
  imagelist = [f for f in os.listdir('./1') if f.endswith('.jpg')]
  print(imagelist)
