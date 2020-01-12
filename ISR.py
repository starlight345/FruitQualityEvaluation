# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 23:25:57 2020

@author: Tanya
"""

#import numpy as np
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
#import matplotlib.image as mpimg 
#import matplotlib.pyplot as plt
import numpy
from PIL import Image 
import glob
import IPython.display as display

path = r'C:\\Users\\Tanya\\Desktop\\IASR\\rottenoranges'
outpath = 'C:\\Users\\Tanya\\Downloads\\AI3\\ISR\\rottenoranges'
outpath2 = 'C:\\Users\\Tanya\\Downloads\\AI3\\ISR\\withoutCorners'

filenames = glob.glob(path + "/*.png") #read all files in the path mentioned
f = glob.glob(outpath + "/*.png")
fpath = glob.glob(outpath2 + "/*.png")

"""
for x in filenames:
    basewidth = 340
    img = Image.open(x)
    #wpercent = (basewidth/float(img.size[0]))
    #hsize = int((float(img.size[1])*float(wpercent)))
    hsize = 277
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    #plt.imshow(img)
    src_fname, ext = os.path.splitext(x)  # split filename and extension
    # construct output filename, basename to remove input directory
    save_fname = os.path.join(outpath, os.path.basename(src_fname)+'.png')
    img.save(save_fname)
"""   
    
   
def changeColorOfEdges(img):
    img = Image.open(img)
    np_im = numpy.array(img)
    #img = mpimg.imread(img)
    #img = Image.open(img)
    # height, width, number of channels in image
    height = img.size[1]
    width = img.size[0]
    for row in range(height): #each pixel has coordinates
        for col in range(width):
            if  (np_im[row, col] <= (0, 0, 0)).all():
                if row < 70 and col < 250:
                    np_im[row, col] = [255]
                elif row < 190 and col > 300:
                    np_im[row, col] = [255]
                elif row > 150 and col < 60:
                    np_im[row, col] = [255]
                elif row > 220 and col > 190:
                    np_im[row, col] = [255]
    new_im = Image.fromarray(np_im)
    return new_im

#image_out = Image.new(img.mode,img.size)
#image_out.putdata(img)
#img.save('C:\\Users\\Tanya\\Downloads\\AI3\\ISR\\banana.png') 
#size, resize, if black turn to white
#img = mpimg.imread('C:\\Users\\Tanya\\Desktop\\IASR\\rottenoranges\\rotated_by_15_Screen Shot 2018-06-12 at 11.18.34 PM.png') 

"""
for i in f:
    #img = Image.open(i)
    #img = mpimg.imread(i)
    img = changeColorOfEdges(i)
    src_fname, ext = os.path.splitext(i)  # split filename and extension
    # construct output filename, basename to remove input directory
    save_fname = os.path.join(outpath2, os.path.basename(src_fname)+'.png')
    img.save(save_fname)
    #plt.imshow(img)
    #display.display(img)
"""
for m in fpath:
    image_bgr = cv2.imread(m)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Rectange values: start x, start y, width, height
    rectangle = (3, 10, 320, 235)
    
    # Create initial mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Run grabCut
    cv2.grabCut(image_rgb, # Our image
            mask, # The Mask
            rectangle, # Our rectangle
            bgdModel, # Temporary array for background
            fgdModel, # Temporary array for background
            5, # Number of iterations
            cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    
    # Multiply image with new mask to subtract background
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
    
    # Show image
    plt.imshow(image_rgb_nobg), plt.axis("off")
    plt.show()

            
"""
#img = changeColorOfEdges('C:\\Users\\Tanya\\Downloads\\AI3\\ISR\\banana.png')
img2 = changeColorOfEdges(outpath + "\\" + "rotated_by_15_Screen Shot 2018-06-12 at 11.18.34 PM.png")
# Output Images 
# Output Images 

plt.imshow(img2)
"""
# rotated_by_15_Screen Shot 2018-06-12 at 8.49.20 PM C:\Users\Tanya\Desktop
# myarray = np.loadtxt(x, skiprows=9)
   # im = Image.fromarray(myarray)
   # im.save(outpath + '/*.tif')
#image_list = []
#for filename in glob.glob('C:\\Users\\Tanya\\Desktop\\IASR\\rottenoranges/*.png'): #assuming gif
   
     
    #image_list.append(im)
