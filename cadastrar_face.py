import cv2
import face_recognition
import os
import numpy as np

if not os.path.exists("rostos"):
  os.makedirs("rostos")
  
  
nome = input("Digite o nome da pessoa: ").strip()

webcam = cv2.VideoCapture(0)
print("Posicione o rosto na câmera. Pressione 's' para salvar.")

while True:
  ret, frame = webcam.read()
  if not ret:
    break
  
  cv2.imshow("Cadastro - Pressione 's' para salvar", frame)
  
  key = cv2.waitKey(1)
  if key == ord('s'):
    
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
      face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
      np.save(f"rostos/{nome}.npy", face_encoding)
      print(f"Rosto de {nome} capturado!")
      break
    
    else:
      print("[!] Nenhum rosto detectado.")
      
  elif key == 27:
    break
  
webcam.release()
cv2.destroyAllWindows()        