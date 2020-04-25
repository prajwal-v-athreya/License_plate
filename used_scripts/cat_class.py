import pandas as pd
import cv2


img = cv2.imread('/sample/NewYork.jpg')
img = cv2.rectangle(img, (314,306),(331,339),(0,255,0),2,2)
cv2.imshow('img',img)
cv2.waitKey(0)
