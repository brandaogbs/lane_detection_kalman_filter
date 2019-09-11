from __future__ import division

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

from log import *


# leitura do vídeo
cap = cv2.VideoCapture('../videos/01.mp4')

# limiares da RoI
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

topLeftPt = (0, height*(3.1/5))
topRightPt = (width, height*(3.1/5))

# montagem da RoI
region_of_interest_points = [(0, height),
                            topLeftPt,
                            topRightPt,
                            (width, height)]

# limiares de cores
min_val_y = np.array([15,80,190])
max_val_y = np.array([30,255,255])
min_val_w = np.array([0,0,195])
max_val_w = np.array([255, 80, 255])

# variavel medir o tempo de cada iteracao
ticks = 0

lt = LaneTracker(2, 0.1, 15)
ld = LaneDetector(100)

while True:
    # atualiza tempo anterior
    precTick = ticks
    
    # inicia novo tick
    ticks = cv2.getTickCount()

    # calcula o delta tempo (tempo atual-anterior)
    dt = (ticks - precTick) / cv2.getTickFrequency()
    
    ret, frame = cap.read()
    if ret:

        cv2.imwrite('../out/original.png', frame)

        # aplica correção de gamma
        gamma = gamma_correction_auto(frame,equalizeHist = False) #0.2
        cv2.imwrite('../out/gamma.png', gamma)

        # extrai a RoI com correção de gamma        
        cropped = region_of_interest(gamma, np.array([region_of_interest_points], np.int32))
        cv2.imwrite('../out/cropped.png', cropped)
        
        # aplica filtro bilateral na RoI
        bilateral = cv2.bilateralFilter(cropped, 9, 80, 80)
        cv2.imwrite('../out/bilateral.png', bilateral)
        
        # aplica filtro hsv com os threshold na RoI
        hsv = hsv_filter(bilateral, min_val_y, max_val_y,  min_val_w, max_val_w)
        cv2.imwrite('../out/hsv.png', hsv)
        
        # faz o predicao da primeira iteracao
        predicted = lt.predict(dt)

        # faz a deteccao das faixas
        lanes = ld.detect(bilateral)
        
        # logDet = "Detected:\nL: {}\nR: {}\n".format(lanes[0], lanes[1])
        if lanes[0] == None:
            logDet = 'D,None,None,None,None'        
        else:
            logDet = "D,"+str(lanes[0]).replace('(', '')
            logDet = logDet.replace(')', '')

        # img auxiliar para ipm        
        helper = np.zeros_like(frame)

        # caso seja o primeiro frame
        if predicted is not None:
            logPre = "P,{},{},{},{}".format(float(predicted[0][0]), float(predicted[0][1]), float(predicted[0][2]), float(predicted[0][3]))
            
            cv2.line(helper, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 255, 0), 2)
            cv2.line(helper, (predicted[1][0], predicted[1][1]), (predicted[1][2], predicted[1][3]), (0, 255, 0), 2)
            
            # deixa a ipm preta
            helper[:int(helper.shape[0]*0.55),:] = 0
            
            # junta o fundo preto com o frame
            frame = cv2.add(helper,frame)

            # monta a ipm
            ipmout = IPM(helper,region_of_interest_points)
            
            # faz update do kalman
            lt.update(lanes)
            
            # mostra a deteccao e o ipmc1
            cv2.imshow('final', frame)
            cv2.imshow('IPM', ipmout)

        else:
            logPre ="P,None,None,None,None"

            # deixa a ipm preta
            helper[:int(helper.shape[0]*0.55),:] = 0
            
            # junta o fundo preto com o frame
            frame = cv2.add(helper,frame)

            # monta a ipm
            ipmout = IPM(helper,region_of_interest_points)
            
            # faz update do kalman
            lt.update(lanes)
            
            # aplica canny na HSV
            canny = cv2.Canny(hsv, 80, 255) #100
            # cv2.imwrite('../out/canny.png', canny)
            
            # aplica transformada de hough linear
            hough, lines = hough_transform(frame, canny, 11, discard_horizontal = 0.7) #14 0.4
            # cv2.imwrite('hough.png', lines)
                        
            final = clustering(lines, frame, np.array([region_of_interest_points], np.int32), eps = 0.5, min_samples = 4)
 
            # mostra a deteccao e o ipmc1
            cv2.imshow('final', frame)
            cv2.imshow('IPM', ipmout)

        # logger.debug(logPre)
        # logger.debug(logDet)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
