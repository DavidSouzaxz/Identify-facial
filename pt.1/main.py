import mediapipe as mp
import os
import numpy as np
import face_recognition
import cv2


webcam = cv2.VideoCapture(0)

solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True: 

 # LER AS INFORMAÇÕES DA WEBCAM
  verificador, frame = webcam.read()
  if not verificador:
    break

  # RECONHECER OS ROSTOS QUE TEM ALI DENTRO
  lista_rostos = reconhecedor_rostos.process(frame)
  
  if lista_rostos.detections:
    for rosto in lista_rostos.detections:
      # DESENHAR OS ROSTOS NA IMAGEM
      desenho.draw_detection(frame, rosto)
  
  cv2.imshow("Face in webcam", frame)

  # QUANDO APERTAR ESC, PARA O LOOP
  if cv2.waitKey(5) == 27: 
    break


webcam.release()