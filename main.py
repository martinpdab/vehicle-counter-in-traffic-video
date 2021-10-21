from __future__ import print_function
import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX

#inisiasi backsub dan video
backSub = cv.createBackgroundSubtractorMOG2()
backSub.setVarThreshold(150)
capture = cv.VideoCapture('vtest.avi')
error = 10
matches = []
jmlMobil = 0
pjgGaris = 164

#nyari centroid
def nilaiCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx, cy


while True:
    #potong video
    ret, frame = capture.read()
    crop = frame[90:450,160:400]
    cropedit = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    #backsub
    fgMask = backSub.apply(cropedit)
    #threshold
    ret,thresh_img = cv.threshold(fgMask,180,255,cv.THRESH_BINARY)
    #morpho filter dilated
    dilated = cv.dilate(thresh_img,np.ones((8,8)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    #morpho filter closing
    closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
    #contour
    con, hir = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for(i, c) in enumerate(con):
        (x, y, w, h) = cv.boundingRect(c)
        contour = (w >= 35) and (h >= 35)

        if not contour:
            continue
        cv.rectangle(crop, (x-10, y-10), (x+w+10, y+h+10), (0,0,255),2)
        cv.line(crop, (55,pjgGaris), (200,pjgGaris), (0,255,0), 3)
        centroid = nilaiCentroid(x, y, w, h)
        matches.append(centroid)
        cv.circle(crop, centroid, 5, (0,0,255), -1)
        cx, cy= nilaiCentroid(x, y, w, h)
        for (x, y) in matches:
            if (y<(pjgGaris + error) and y >(pjgGaris - error)) and ((x > 50 - error) and (x < 200 + error)):
                jmlMobil += 1
                matches.remove((x,y))
                print(jmlMobil)

    print(jmlMobil)
    
    


    cv.imshow('Frame', frame)
    cv.imshow('Cropped', crop)
    cv.imshow('Backsub', fgMask)
    cv.imshow('Dilated', dilated)
    cv.imshow('Closing FIlter', closing)

    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break



