import cv2

img = cv2.imread('../dataset/images/img_0000004.jpg')
cv2.imshow('ori',img)
img1 = cv2.flip(img,1)
cv2.imshow('1',img1)
img0 = cv2.flip(img,0)
cv2.imshow('0',img0)
cv2.waitKey(0)