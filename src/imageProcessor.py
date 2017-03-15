import cv2
import numpy as np
import glob
import os

def setDefaultImages():

   images = glob.glob('images/*')

   for item in images:
      #read in images
      img = cv2.imread(item)
      #resize images
      res = cv2.resize(img,(100, 100), interpolation = cv2.INTER_CUBIC)
      #grey scale images
      res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

      #split path name into images/... and .jpeg
      filename, file_extension = os.path.splitext(item)
      #split file name by '/' to extract name of jpeg
      arr = filename.split('/')
      l = len(arr)
      #name new base images and put in startImages
      name = "startImages/" + arr[l-1] + file_extension
      #write images to file
      cv2.imwrite(name, res)

def rotateImages(imageList):

   for item in imageList:
      filename, file_extension = os.path.splitext(item)

      img = cv2.imread(item)
      rows,cols = img.shape[:2]
      angle = 0
      for angle in range(1, 4):
         M = cv2.getRotationMatrix2D((50,50),angle * 90,1)
         writeTo = cv2.warpAffine(img,M,(rows,cols))
         name = filename + str(angle) + file_extension
         cv2.imwrite(name, writeTo)

def blurImages(imageList):
   for item in imageList:
      filename, file_extension = os.path.splitext(item)
      img = cv2.imread(item)
      blur = cv2.GaussianBlur(img,(1,1),0)  
      name = filename + "-Blur1" + file_extension 
      cv2.imwrite(name, blur)   
      blur = cv2.GaussianBlur(img,(3,3),0)  
      name = filename + "-Blur2" + file_extension 
      cv2.imwrite(name, blur)   
      blur = cv2.GaussianBlur(img,(5,5),0)  
      name = filename + "-Blur3" + file_extension 
      cv2.imwrite(name, blur)  
      blur = cv2.GaussianBlur(img,(7,7),0)  
      name = filename + "-Blur4" + file_extension 
      cv2.imwrite(name, blur)
      blur = cv2.GaussianBlur(img,(9,9),0)  
      name = filename + "-Blur5" + file_extension 
      cv2.imwrite(name, blur) 

def equalize(imageList):
   for item in imageList:
      filename, file_extension = os.path.splitext(item)
      img = cv2.imread(item, 0) #load as grayscale?
      dst = cv2.equalizeHist(img);
      name = filename + "-Equalized" + file_extension 
      cv2.imwrite(name, dst) 

def erosion(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      erosion = cv2.erode(img,kernel,iterations = 1)
      name = filename + "-Eroded" + file_extension
      cv2.imwrite(name, erosion)

def dilation(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      dilation = cv2.dilate(img,kernel,iterations = 1)
      name = filename + "-Dilated" + file_extension
      cv2.imwrite(name, dilation)

def morphOpen(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
      name = filename + "-MorphedOpen" + file_extension
      cv2.imwrite(name, morph)

def morphClose(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
      name = filename + "-MorphedClose" + file_extension
      cv2.imwrite(name, morph)

def morphGradient(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
      name = filename + "-MorphedGradient" + file_extension
      cv2.imwrite(name, morph)

def morphTophat(imageList):
   for item in imageList:
      img = cv2.imread(item)
      filename, file_extension = os.path.splitext(item)
      kernel = np.ones((5,5),np.uint8)
      morph = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
      name = filename + "-MorphedTophat" + file_extension
      cv2.imwrite(name, morph)

def main():
   setDefaultImages()
   images = glob.glob('startImages/*')
   rotateImages(images)
   images = glob.glob('startImages/*')
   blurImages(images)
   equalize(images)
   erosion(images)
   dilation(images)
   morphOpen(images)
   morphClose(images)
   morphGradient(images)
   morphTophat(images) #could be too much

if __name__ == "__main__":
    main()
