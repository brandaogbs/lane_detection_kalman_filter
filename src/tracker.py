import cv2
import numpy as np
import math

from scipy.linalg import block_diag

class LaneTracker:
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale, process_cov_parallel=0, proc_noise_type='white'):
        
        # def dos estados utilizados no estimador        
        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes
        self.state_size = self.meas_size * 2
        self.contr_size = 0

        # def do filtro de kalman baseado nos estados
        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.contr_size)

        # matriz de transiscao (A)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)

        # matriz de medidas (H)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size), np.float32)

        for i in range(self.meas_size):
            self.kf.measurementMatrix[i, i*2] = 1

        # monta o ruido de processo (matrix Q)
        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5], [0.5, 1.]], dtype=np.float32)
            self.kf.processNoiseCov = block_diag(*([block] * self.meas_size)) * proc_noise_scale
        if proc_noise_type == 'identity':
            self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * proc_noise_scale
        
        # monta a matriz Q
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                self.kf.processNoiseCov[i, i+(j*8)] = process_cov_parallel
                self.kf.processNoiseCov[i+(j*8), i] = process_cov_parallel

        # matriz R        
        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * meas_noise_scale

        # matriz de erro a priori
        self.kf.errorCovPre = np.eye(self.state_size)

        # zera a matriz de medidas e de estados
        self.meas = np.zeros((self.meas_size, 1), np.float32)
        self.state = np.zeros((self.state_size, 1), np.float32)

        # atualiza a flag de primeiro quadro (sig. que esta na etapa de estimacao)
        self.first_detected = False

    # update temporal
    def _update_dt(self, dt):
        # atualiza A com dt
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt

    def _first_detect(self, lanes):
        # atualiza os estados com as pos. das faixas detectadas
        for l, i in zip(lanes, range(0, self.state_size, 8)):
            self.state[i:i+8:2, 0] = l
        
        # atualiza estado a posteriore com estado atual
        self.kf.statePost = self.state

        # atualiza flag de primeira detecção
        self.first_detected = True

    def update(self, lanes):

        # caso seja a primeira deteccao
        if self.first_detected:
            # atualiza o vetor de medidas com as posicoes das faixas
            for l, i in zip(lanes, range(0, self.meas_size, 4)):
                if l is not None:
                    self.meas[i:i+4, 0] = l

            # atualiza o kalman com as medidas
            self.kf.correct(self.meas)
        else:
            if lanes.count(None) == 0:
                self._first_detect(lanes)

    def predict(self, dt):

        # no caso da primeira deteccao estima os estados do kalman
        if self.first_detected:
            # atualizacao temporal
            self._update_dt(dt)

            # faz predicao dos estados do kalman
            state = self.kf.predict()

            # retorna o list com os estados (pos. das faixas)
            lanes = []
            for i in range(0, len(state), 8):
                lanes.append((state[i], state[i+2], state[i+4], state[i+6]))
            return lanes

        # caso nao seja a primeira deteccao retorna None
        else:
            return None