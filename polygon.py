from shapely.geometry import Polygon,LineString,Point
import numpy as np

# p1 = Polygon([(0,0), (1,1), (1,0)])
# p2 = Polygon([(0,1), (1,0), (1,1)])
# p1.intersection(p2).area

polygon = [(0,0), (1,0), (2,1), (1,2), (0,1)]

pol2 = np.array([1,1]) + np.array([(0,0), (1,0), (2,1), (1,2), (0,1)])
shapely_poly = Polygon(polygon)
print(Point(shapely_poly.exterior.coords[0]))
sh_poly2 = Polygon(pol2)
int_poly = shapely_poly.union(sh_poly2)
print(int_poly)
line = [(-1, -1), (2.0, 2)]
shapely_line = LineString(line)

intersection_line = list(shapely_poly.intersection(shapely_line).coords)
print(intersection_line)

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

def caculate_image_metric(w,h,poly):
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

def get_quad(w,h,H):
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

# def get_new_poly(poly,quad):
#     poly.union(quad)

def intersect_poly_quad(poly, quad):
    shapely_poly = Polygon(poly)
    shapely_quad = Polygon(quad)
    intersections = []
    pt1 = quad[0]
    for point in quad[1:]:
        shapely_line = LineString(pt1,point)
        intersections.extend(list(shapely_poly.intersection(shapely_line).coords))
    
    #  get boundary line
    found = False
    for pt1 in intersections:
        if (found):
            break
        for pt2 in intersections:
            if (found):
                break
            p = Point((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2)
            if (shapely_poly.contains(p) and shapely_quad.contains(p)):
                inter_line =  (pt1,pt2)
                found = True
                
