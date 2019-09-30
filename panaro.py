import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from random import randrange
import os
from math import sqrt
from shapely.geometry import Polygon,LineString,Point

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

