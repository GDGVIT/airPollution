import numpy as np
import cv2
from PIL import Image
import urllib.parse
import urllib.request
import io
from math import log, exp, tan, atan, pi, ceil

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0


def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi / 360.0)) / (pi / 180.0)
    my = (my * ORIGIN_SHIFT) / 180.0
    res = INITIAL_RESOLUTION / (2 ** zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py


def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2 ** zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180.0)) - pi / 2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon


upperleft = '12.92,79.11'
lowerright = '12.91,79.13'

zoom = 18

ullat, ullon = map(float, upperleft.split(','))
lrlat, lrlon = map(float, lowerright.split(','))


scale = 1
maxsize = 640


ulx, uly = latlontopixels(ullat, ullon, zoom)
lrx, lry = latlontopixels(lrlat, lrlon, zoom)


dx, dy = lrx - ulx, uly - lry


cols, rows = int(ceil(dx / maxsize)), int(ceil(dy / maxsize))


bottom = 120
largura = int(ceil(dx / cols))
altura = int(ceil(dy / rows))
alturaplus = altura + bottom

final = Image.new("RGB", (int(dx), int(dy)))
for x in range(cols):
    for y in range(rows):
        dxn = largura * (0.5 + x)
        dyn = altura * (0.5 + y)
        latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom / 2, zoom)
        position = ','.join((str(latn), str(lonn)))
        print(x, y, position)
        urlparams = urllib.parse.urlencode({'center': position,
                                            'zoom': str(zoom),
                                            'size': '%dx%d' % (largura, alturaplus),
                                            'maptype': 'satellite',
                                            'sensor': 'false',
                                            'scale': scale,
                                            'key': 'AIzaSyDyjYNDolk6JSa0two3Z5ctru20tlpvkSg'})
        url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
        f = urllib.request.urlopen(url)
        image = io.BytesIO(f.read())
        im = Image.open(image)
        im.save("map.png")


        img = cv2.imread('map.png')
        shifted = cv2.pyrMeanShiftFiltering(img,7,30)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(shifted,cv2.COLOR_BGR2HSV)

        lower_trees = np.array([10,0,10])
        higher_trees = np.array([180,180,75])

        lower_houses = np.array([90,10,100])
        higher_houses = np.array([255,255,255])

        lower_roads = np.array([90,10,100])
        higher_roads = np.array([100,100,100])

        lower_feilds = np.array([0,20,100])
        higher_feilds = np.array([50,255,255])

        lower_feilds_blue = np.array([0,80,100])
        higher_feilds_blue = np.array([255,250,255])


        masktree = cv2.inRange(hsv,lower_trees,higher_trees)
        maskhouses = cv2.inRange(hsv,lower_houses,higher_houses)
        maskroads = cv2.inRange(hsv,lower_roads,higher_roads)
        maskfeilds_houses = cv2.inRange(hsv,lower_feilds,higher_feilds)
        blue_limiter = cv2.inRange(hsv,lower_feilds_blue,higher_feilds_blue)
        maskfeilds = maskfeilds_houses
        res = cv2.bitwise_and(img,img,mask=maskfeilds)



        # show the output image
        cv2.imshow('res',res)
        #cv2.imshow('mask',maskfeilds)
        cv2.imshow('img', img)
        #cv2.imshow("hsv", hsv)
        cv2.waitKey(delay=2000)
        cv2.destroyAllWindows()
