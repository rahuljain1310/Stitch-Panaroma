import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from random import randrange
import os
from math import sqrt
from shapely.geometry import Polygon,LineString,Point

def get_keypoints(img1,img2):
  kp1,des1 = sift.detectAndCompute(img1,None)
  kp2,des2 = sift.detectAndCompute(img2,None)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)
def is_inside_image(img,i,j):
  if (i<img.shape[0] and j<img.shape[1]):
    # return img[i][j][0] > 0
   
    return (img[i][j][0] > 0 or img[i][j][1] > 0 or img[i][j][2] > 0)
  else:
    return False

def get_line(p1,p2):
  a = p2[1]-p1[1]
  b = p1[0]-p2[0]
  c = p2[0]*p1[1]-p1[0]*p2[1]
  dist = sqrt(a*a + b*b)

  return (a/dist,b/dist,c/dist)
def find_metric(point,poly,dists):
    p = Point(point)
    poly_coords = poly.exterior.coords

    # points = 
    metrics = []
    pt1 = Point(poly_coords[0])
    dis1 = p.distance(pt1)
    for point,dist in zip(poly_coords[1:],dists):
        dis2 = p.distance(point)
        metrics.append(abs(dis2+dis1-dist))
        dis1 = dis2
    return min(metrics)

def calculate_dists(poly):
    poly_coords = poly.exterior.coords
    dists = []
    pt1 = Point(poly_coords[0])
    for point in poly_coords[1:]:
        dists.append(pt1.distance(point))
        pt1 = Point(point)
    return dists

def calculate_image_metric(w,h,poly):
    dists = calculate_dists(poly)
    image_metrics = np.ndarray([h,w],dtype=np.float32)
    for i in range(w):
        for j in range(h):
            image_metrics[j][i] = find_metric(Point(i,j),poly,dists)
    return image_metrics

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
  for i in range(h):
    for j in range(w):
      if (i>=ty and j >=tx):
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
  dst = np.array(img1)
  assert(img1.shape == img2.shape)
  for i in img1.shape[0]:
    for j in img1.shape[1]:
      if (mask1[i][j] and mask2[i][j]):
        for k in range(3):
          dst[i][j][k] = np.uint8(min((metric1[i][j]*img1[i][j][k] + metric2[i][j]*img2[i][j][k])/(metric1[i][j] + metric2[i][j]),255))
        # dst[i][j] = np.array((metric1[i][j]*img1[i][j] + metric2[i][j]*img2[i][j])/(metric1[i][j] + metric2[i][j]),
      elif (mask1[i][j]):
        dst[i][j] = img1[i][j].copy()
      elif (mask2[i][j]):
        dst[i][j] = img2[i][j].copy()
      else:
        dst[i][j] = np.array([0,0,0],dtype=np.uint8)
  return dst

def combine(img1,H1,img2,H2,w,h,mask1,poly):
    img_1 = cv2.warpPerspective(img1,H1,(w,h))
    img_2 = cv2.warpPersepctive(img2,H2,(w,h))
    tx = H1[0][2]
    ty = H1[1][2]
    mask1_new = get_new_mask(mask1,tx,ty,h,w)
    quad = get_quad(img2.shape[1],img2.shape[0],H2)
    mask2 = get_mask(w,h,quad)
    mask_new = mask1_new + mask2
    poly1_new = shift_poly(poly)
    poly_new = poly1_new.union(quad)
    img1_metric = calculate_image_metric(w,h,poly1_new)
    img2_metric = calculate_image_metric(w,h,quad)
    dst = blend(img1,mask1_new,img1_metric,img2,mask2,img2_metric)
    return (dst,mask_new,poly_new)
  
def panaroma(imglist,Hpairs,wh):
    img1 = imglist[0]
    mask1 = np.ones([img1.shape[0],img1.shape[1]],dtype=np.bool)
    poly1 = Polygon((0,0),(img1.shape[1],0),(img1.shape[1],img1.shape[0]),(0,img1.shape[0]))

    for i in range(Hpairs):
      H1,H2 = Hpairs[i]
      w,h = wh[i]
      img2 = imglist[i+1]
      img1,mask1,poly1 = combine(img1,H1,img2,H2,w,h,mask1,poly1)
    
    return img1
    
# def get_mask(img,corners):
#   mask = np.ndarray([img.shape[0],img.shape[1]],dtype = np.bool)
#   p =  path.Path(corners[0])
#   is_inside = lambda t: p.contain_points(t)
#   for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#       mask[i][j] = p.contains_points([(j,i)])[0]
#       # print(mask[i][j])
#   return mask
#   # if (p.contain_points(img))
# # def blending(img1,img2):
# def get_lines(width ,height, H):
#   four_points = [0,0,0,0]
#   print(H)
#   four_points[0] = np.array([0,0,1])
#   four_points[1] = np.array([width-1,0,1])
#   four_points[2] = np.array([width-1,height-1,1])
#   four_points[3] = np.array([0,height-1,1])
#   pt1 = None
#   lines = []
#   for i in range(4):
#     x =  np.matmul(H,four_points[i])
#     x = [x[0]/x[2],x[1]/x[2]]
#     if pt1 is not None:
#       lines.append(get_line(pt1,x)) 
#     pt1 = x
#     four_points[i] = x
#   x = four_points[0]
#   lines.append(get_line(pt1,x))
#   print(four_points)
#   return (four_points,lines)
   

def laplacian_blending(img1,img2,mask1,H,w,h):
  corners = get_lines(w,h,H)
  mask2 = get_mask(img2,corners)
  # plt.imshow(mask2)
  # plt.show()
  # print(mask2.shape)
  dest = Laplacian_Pyramid_Blending_with_mask(img1,img2,mask1,mask2)
  mask_dest = mask1+mask2
  return dest,mask_dest

def Laplacian_Pyramid_Blending_with_mask(A, B, m1,m2, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = np.ndarray([m1.shape[0],m1.shape[1],3],dtype=np.float32)
    GM2 = np.ndarray([m2.shape[0],m2.shape[1],3],dtype= np.float32)

    for i in range(GM.shape[0]):
      for j in range(GM.shape[1]):
        GM[i][j] = np.array([m1[i][j],m1[i][j],m1[i][j]],dtype=np.float32)
        GM2[i][j]= np.array([m2[i][j],m2[i][j],m2[i][j]],dtype=np.float32)
    
    for i in range(m2.shape[0]):
      for j in range(m2.shape[1]):
        for k in range(3):
          GM2[i][j][k] = max(0,GM2[i][j][k]-GM[i][j][k])

    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    gpM2 = [GM2]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        GM2 = cv2.pyrDown(GM2)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
        gpM2.append(np.float32(GM2))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpM1r = [gpM[num_levels-1]]
    gpM2r = [gpM2[num_levels-1]]

    # gpM2 = [gpM2[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)  
        gpM1r.append(gpM[i-1]) 
        gpM2r.append(gpM2[i-1])
        # gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm,gm2 in zip(lpA,lpB,gpM1r,gpM2r):
        ls = la * gm + lb * gm2
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

# if __name__ == '__main__':
#     A = cv2.imread("input1.png",0)
#     B = cv2.imread("input2.png",0)
#     m = np.zeros_like(A, dtype='float32')
#     m[:,A.shape[1]/2:] = 1 # make the mask half-and-half
#     lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 5)
#     cv2.imwrite("lpb.png",lpb)

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
# dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]+img_.shape[0]))
dst = cv2.warpPerspective(img_,H,(1024,1024))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
img_t1 = np.zeros_like(dst, dtype=np.uint8)
mask1 = np.zeros([dst.shape[0],dst.shape[1]],dtype=np.uint8)
img_t1[0:img.shape[0],0:img.shape[1]] = img
mask1[0:img.shape[0],0:img.shape[1]] = 1
print(mask1.shape, dst.shape, img_t1.shape)
dst1,mkdst = laplacian_blending(img_t1,dst,mask1,H,img2.shape[1],img2.shape[0])


# dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('output.jpg',dst1)
plt.imshow(dst)
plt.show()

print(matches.shape)

if __name__ == "__main__":
  ## Read all Images
  imagelist = [f for f in os.listdir('./1') if f.endswith('.jpg')]
  print(imagelist)
