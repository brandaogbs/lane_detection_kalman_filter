# -*- coding: utf-8 -*-
import time
import math

import cv2
import numpy as np

from tracker import LaneTracker
from detector import LaneDetector   
from processor import region_of_interest, clustering, gamma_correction, gamma_correction_auto, hough_transform, hsv_filter, IPM


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

# declara o tracker
lt = LaneTracker(2, 0.1, 15)

# declara o detector
ld = LaneDetector(100)

while True:
    # atualiza tempo anterior
    precTick = ticks
    
    # inicia novo tick
    ticks = cv2.getTickCount()

    # calcula o delta tempo (tempo atual-anterior)
    dt = (ticks - precTick) / cv2.getTickFrequency()
 
    # faz leitura do frame
    ret, frame = cap.read()
    if ret:
        # aplica correção de gamma
        gamma = gamma_correction_auto(frame,equalizeHist = False) #0.2
        
        # extrai a RoI com correção de gamma
        cropped = region_of_interest(gamma, np.array([region_of_interest_points], np.int32))
        
        # aplica filtro bilateral na RoI
        bilateral = cv2.bilateralFilter(cropped, 9, 80, 80)
        
        # aplica filtro hsv com os threshold na RoI
        hsv = hsv_filter(cropped, min_val_y, max_val_y,  min_val_w, max_val_w)
       
        # faz o predicao da primeira iteracao
        predicted = lt.predict(dt)

        # faz a deteccao das faixas
        lanes = ld.detect(cropped)
                
        # img auxiliar para ipm        
        helper = np.zeros_like(frame)
        
        # caso seja o primeiro frame
        if predicted is not None:
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
        
        # aplica canny na HSV
        canny = cv2.Canny(hsv, 80, 255) #100
        
        # aplica transformada de hough linear
        hough, lines = hough_transform(frame, canny, 11, discard_horizontal = 0.7) #14 0.4
        
        _, frame = cap.read()
        
        final = clustering(lines, frame, np.array([region_of_interest_points], np.int32), eps = 0.5, min_samples = 4)
        
        # mostra a deteccao e o ipm
        cv2.imshow('final', frame)
        cv2.imshow('IPM', ipmout)

        # escape 'q'        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()