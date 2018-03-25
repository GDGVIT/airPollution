import numpy as np
'''

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift,estimate_bandwidth
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle
from PIL import Image
import cv2
pylab.rcParams['figure.figsize']=16,12
image=Image.open('images[464].jpg')
image=np.array(image)
original_shape=image.shape
print(len(image))

flat_image=np.reshape(image,[-1,3])
#flat_image=np.delete(flat_image,3,1)
#print(flat_image)
plt.imshow(image)
bandwidth=estimate_bandwidth(flat_image,quantile=.2,n_samples=10000)

ms=MeanShift(bandwidth,bin_seeding=True)
ms.fit(flat_image)
labels=ms.labels_
cluster_centers=ms.cluster_centers_
labels_unique=np.unique(labels)
n_clusters_=len(labels_unique)
print(n_clusters_)

segmented_image=np.reshape(labels,original_shape[:2])
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.axis('off')
plt.show()


========================================================

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)

    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    #cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#from skimage.feature import peak_local_max
#from skimage.morphology import watershed
from scipy import ndimage
import argparse
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
'''
import cv2



img = cv2.imread('mapsharan[576].png')
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
higher_feilds = np.array([255,60,255])

lower_feilds_blue = np.array([0,80,100])
higher_feilds_blue = np.array([255,250,255])


masktree = cv2.inRange(hsv,lower_trees,higher_trees)
maskhouses = cv2.inRange(hsv,lower_houses,higher_houses)
maskroads = cv2.inRange(hsv,lower_roads,higher_roads)
maskfeilds_houses = cv2.inRange(hsv,lower_feilds,higher_feilds)
blue_limiter = cv2.inRange(hsv,lower_feilds_blue,higher_feilds_blue)
maskfeilds = maskfeilds_houses - maskhouses - blue_limiter
res = cv2.bitwise_and(img,img,mask=maskfeilds)



# show the output image
cv2.imshow('res',res)
cv2.imshow('mask',maskfeilds)
cv2.imshow('img', img)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()