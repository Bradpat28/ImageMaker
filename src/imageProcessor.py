import cv2
import numpy as np

img = cv2.imread('Fruit.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray_fruit.jpeg', gray)

cv2.imshow('img', img)
cv2.imshow('gray', gray)

cv2.waitKey(0) 
cv2.destroyAllWindows()