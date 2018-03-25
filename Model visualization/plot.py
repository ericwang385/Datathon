# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap
import pickle 
import cv2
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np



with open('ground_truth.pkl','rb') as f:
     inf = pickle.load(f)

     
'''date = np.log(inf[n].pick_up + 1)
date = Image.fromarray(date)
date = np.asarray(date.resize((552,546),Image.ANTIALIAS))
index = np.where(date>1)
img[index[0],index[1],3]=0'''
#print(date.min())
#plt.imshow(img)
#plt.imshow(date,cmap='Reds')
#img = img.resize((200,200),Image.ANTIALIAS)
#plt.ion()


for n in range(168):
    fig = plt.figure(figsize=(16,6), frameon = False )
    img = cv2.imread('map.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    date = inf[n].drop_off
    date = date.clip(min=0)
    date = np.log(date+ 1)
    print(date.min())
    date = Image.fromarray(date)
    date = np.asarray(date.resize((552,546),Image.ANTIALIAS))
    index = np.where(date>=1)
    img[index[0],index[1],3]=0
    plt.imshow(date,cmap='Blues',vmin=0,vmax=6)
    plt.colorbar()
    plt.imshow(img)
    plt.axis("off")
    plt.title('Ground_truth 2013-%d-%d  %d:00'%(inf[n].date_hour.month,inf[n].date_hour.day,inf[n].date_hour.hour))
    plt.savefig('Ground_truth 2013-%d-%d  %d:00.png'%(inf[n].date_hour.month,inf[n].date_hour.day,inf[n].date_hour.hour))
    plt.pause(0.1)
    plt.close()
plt.ioff()
plt.show()



        
