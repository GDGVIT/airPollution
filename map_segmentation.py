import numpy as np
import cv2
from PIL import Image
import urllib.parse
import urllib.request
import io
from math import log, exp, tan, atan, pi, ceil
from place_lookup import find_coordinates
from calc_area import afforestation_area

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

query = input('What kinda places you want me look up? ')
results = find_coordinates(query)


zoom = 18

ullat, ullon = results['upper_left']
lrlat, lrlon = results['lower_right']


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
                                            'key': 'AIzaSyA_d4uV3HqPPWbCb77VhXNYn5UcXRLAiVc'})
        urlparamsmaps = urllib.parse.urlencode({'center': position,
                                                'zoom': str(zoom),
                                                'size': '%dx%d' % (largura, alturaplus),
                                                'maptype': 'roadmap',
                                                'sensor': 'false',
                                                'scale': scale,
                                                'key': 'AIzaSyA_d4uV3HqPPWbCb77VhXNYn5UcXRLAiVc'})
        url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
        url1 = 'http://maps.google.com/maps/api/staticmap?' + urlparamsmaps
        f = urllib.request.urlopen(url)
        h = urllib.request.urlopen(url1)
        image = io.BytesIO(f.read())
        imagemaps = io.BytesIO(h.read())
        im = Image.open(image)
        immaps = Image.open(imagemaps)
        im.save("map.png")
        immaps.save("map_normal.png")

        img = cv2.imread('map.png')
        img_maps = cv2.imread('map_normal.png')
        shifted = cv2.pyrMeanShiftFiltering(img, 7, 30)
        shifted_normal = cv2.pyrMeanShiftFiltering(img_maps, 7, 30)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
        hsv_normal = cv2.cvtColor(shifted_normal, cv2.COLOR_BGR2HSV)

        lower_trees = np.array([10,0,30])
        higher_trees = np.array([180,100,95])

        lower_houses = np.array([90,10,100])
        higher_houses = np.array([255,255,255])

        lower_roads = np.array([0, 0, 250])
        higher_roads = np.array([20, 20, 255])

        lower_feilds = np.array([0,50,100])
        higher_feilds = np.array([50,255,130])

        lower_feilds_blue = np.array([0,80,100])
        higher_feilds_blue = np.array([255,250,255])

        masktree = cv2.inRange(hsv, lower_trees, higher_trees)
        maskhouses = cv2.inRange(hsv, lower_houses, higher_houses)
        maskroads = cv2.inRange(hsv_normal, lower_roads, higher_roads)
        maskfeilds = cv2.inRange(hsv, lower_feilds, higher_feilds)
        gausssion_blur_maskfields = cv2.GaussianBlur(maskfeilds, (15, 15), 0)
        gausssion_blur_masktree = cv2.GaussianBlur(masktree, (15, 15), 0)
        blue_limiter = cv2.inRange(hsv, lower_feilds_blue, higher_feilds_blue)
        res_roads = cv2.bitwise_and(img_maps, img, mask=maskroads)
        #res_houses = cv2.bitwise_and(img,img,mask=maskhouses)
        res_feilds = cv2.bitwise_and(img, img, mask=gausssion_blur_maskfields)
        res_trees = cv2.bitwise_and(img, img, mask=masktree)





        # show the output image
        cv2.imshow('res', res_trees)
        cv2.imshow('res_fields', res_feilds)
        cv2.imshow('res_roads', res_roads)
        #cv2.imshow('res_houses',res_houses)
        #cv2.imshow('mask',maskfeilds)
        cv2.imshow('img', img)
        #cv2.imshow("hsv", hsv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

tot_land_area_acres, number_of_trees = afforestation_area()
