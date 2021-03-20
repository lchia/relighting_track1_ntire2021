import cv2
import numpy as np
print("fusing image")
for i in range(45):
    img1_path = '../TMP/0128/Image%03d.png'%(i+345)
    img2_path = '../TMP/0090/Image%03d.png'%(i+345)
    img3_path = '../TMP/0123/Image%03d.png'%(i+345)
    img4_path = '../TMP/0124/Image%03d.png'%(i+345)
    img5_path = '../TMP/0134/Image%03d.png'%(i+345)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    img4 = cv2.imread(img4_path)
    img5 = cv2.imread(img5_path)
    img1 = img1.astype('float32')
    img2 = img2.astype('float32')
    img3 = img3.astype('float32')
    img4 = img4.astype('float32')
    img5 = img5.astype('float32')
    img = (2*img1+img2+img3+img4+img5)/6
    cv2.imwrite('../results/Image%03d.png'%(i+345), img)
