from __future__ import division

try:
    import cv2
    import math
    import numpy as np
    import time

    from tracker import LaneTracker
    from detector import LaneDetector
    from processor import   (region_of_interest,
                            clustering,
                            gamma_correction, 
                            gamma_correction_auto, 
                            hough_transform, 
                            hsv_filter, 
                            IPM)

    print("Importing.. OK")
except:
    print("Importing.. Error")


#capturing video
cap = cv2.VideoCapture('../videos/01.mp4')

#defining corners for ROI
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

topLeftPt = (0, height*(3.1/5))
topRightPt = (width, height*(3.1/5))

region_of_interest_points = [(0, height),
                            topLeftPt,
                            topRightPt,
                            (width, height)]

#defining color thresholds
min_val_y = np.array([15,80,190])
max_val_y = np.array([30,255,255])
min_val_w = np.array([0,0,195])
max_val_w = np.array([255, 80, 255])

ticks = 0

lt = LaneTracker(2, 0.1, 15)
ld = LaneDetector(100)

while True:

    precTick = ticks
    ticks = cv2.getTickCount()
    dt = (ticks - precTick) / cv2.getTickFrequency()
    ret, frame = cap.read()
    if ret:
        gamma = gamma_correction_auto(frame,equalizeHist = False) #0.2
        cv2.imshow('gamma', gamma)
        cropped = region_of_interest(gamma, np.array([region_of_interest_points], np.int32))
        cv2.imshow('cropped', cropped)
        bilateral = cv2.bilateralFilter(cropped, 9, 80, 80)
        cv2.imshow('bilateral', bilateral)
        hsv = hsv_filter(cropped, min_val_y, max_val_y,  min_val_w, max_val_w)
        predicted = lt.predict(dt)

        lanes = ld.detect(cropped)
                
        helper = np.zeros_like(frame)
        
        if predicted is not None:
            cv2.line(helper, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 255, 0), 2)
            cv2.line(helper, (predicted[1][0], predicted[1][1]), (predicted[1][2], predicted[1][3]), (0, 255, 0), 2)
        
        helper[:int(helper.shape[0]*0.55),:] = 0
        frame = cv2.add(helper,frame)
        ipmout = IPM(helper,region_of_interest_points)
        lt.update(lanes)
        cv2.imshow('hsv', hsv)
        canny = cv2.Canny(hsv, 80, 255) #100
        cv2.imshow('canny', canny)
        hough, lines = hough_transform(frame, canny, 11, discard_horizontal = 0.7) #14 0.4
        cv2.imshow('hough', hough)
        _, frame = cap.read()
        final = clustering(lines, frame, np.array([region_of_interest_points], np.int32), eps = 0.5, min_samples = 4)
        cv2.imshow('final', frame)
        cv2.imshow('IPM', ipmout)
        #out.write(frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()