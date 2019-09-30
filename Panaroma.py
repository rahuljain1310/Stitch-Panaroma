import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from random import randrange
import os
from math import sqrt
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

def get_mask(img,corners):
  mask = np.ndarray([img.shape[0],img.shape[1]],dtype = np.bool)
  p =  path.Path(corners[0])
  is_inside = lambda t: p.contain_points(t)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      mask[i][j] = p.contains_points([(j,i)])[0]
      # print(mask[i][j])
  return mask
  # if (p.contain_points(img))
# def blending(img1,img2):
def get_lines(width ,height, H):
  four_points = [0,0,0,0]
  print(H)
  four_points[0] = np.array([0,0,1])
  four_points[1] = np.array([width-1,0,1])
  four_points[2] = np.array([width-1,height-1,1])
  four_points[3] = np.array([0,height-1,1])
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
  print(four_points)
  return (four_points,lines)
   

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
