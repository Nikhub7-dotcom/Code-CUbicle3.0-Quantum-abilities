"""import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
counter = 0
folder = "D:\\sign\\dataset"

while True:
    success, img = cap.read()  # Corrected to call the read method
    hands, img = detector.findHands(img)
    
    if hands:  # Corrected to check if hands are detected
        hand = hands[0] 
        x, y, w, h = hand['bbox']
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255  # Corrected shape and type
        imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  # Corrected slicing
        imgcropshape = imgcrop.shape
        aspectratio = h / w

        if aspectratio > 1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wgap:wgap + wcal] = imgresize  # Corrected slicing
        else:
            k = imgsize / w  # Corrected calculation
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hgap + hcal, :] = imgresize  # Corrected slicing

        cv2.imshow('imagecrop', imgcrop)
        cv2.imshow('imagewhite', imgwhite)  # Changed window name for clarity

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgwhite)  # Corrected file extension
        print(counter)"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of two hands
offset = 20
imgsize = 300
counter = 0
folder = r"C:\Users\lenovo\OneDrive\Desktop\project\dataset bsl\9"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:  # Check if hands are detected
        imgwhite = np.ones((imgsize, imgsize * 2, 3), np.uint8) * 255  # Adjusted for two hands
        for i, hand in enumerate(hands):
            x, y, w, h = hand['bbox']
            imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            aspectratio = h / w

            if aspectratio > 1:
                k = imgsize / h
                wcal = math.ceil(k * w)
                imgresize = cv2.resize(imgcrop, (wcal, imgsize))
                wgap = math.ceil((imgsize - wcal) / 2)
                imgwhite[:, i * imgsize + wgap:i * imgsize + wgap + wcal] = imgresize
            else:
                k = imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgcrop, (imgsize, hcal))
                hgap = math.ceil((imgsize - hcal) / 2)
                imgwhite[hgap:hgap + hcal, i * imgsize:i * imgsize + imgsize] = imgresize

        cv2.imshow('imagewhite', imgwhite)  # Display combined image

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgwhite)
        print(counter)

