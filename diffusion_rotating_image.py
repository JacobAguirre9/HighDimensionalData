import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pydiffmap import diffusion_map as dm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import cv2
import imutils
from pydiffmap.visualization import embedding_plot, data_plot

img = cv2.imread('gt.png')
img = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
plt.subplot(3, 6, 1), plt.imshow(gray, 'gray')
plt.xlabel("GT @ {degrees} degrees".format(degrees=0))
imgs = np.zeros((18,gray.shape[0]*gray.shape[1]))
imgs[0] = gray.flatten()
rot = gray
# print(imgs.shape)
for r in range(1,18):
    rimg = imutils.rotate(gray, angle=10*r)
    plt.subplot(3, 6, r+1), plt.imshow(rimg, 'gray')
    plt.xlabel("GT @ {degrees} degrees".format(degrees=10*r))
    img_arr = np.asarray(rimg)
    imgs[r] = img_arr.flatten()


neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}

mydmap = dm.DiffusionMap.from_sklearn(n_evecs=1, k=512, epsilon='bgh', alpha=1, neighbor_params=neighbor_params)
# fit to data and return the diffusion map.
dmap = mydmap.fit_transform(imgs)
print(dmap)
plt.figure(0)
plt.scatter(np.arange(0,18)*10, dmap, c=np.arange(0,18)*10, cmap='plasma')
plt.xlabel("orientaion (in degrees)")
plt.ylabel("diffusion coordinate")
plt.show()



