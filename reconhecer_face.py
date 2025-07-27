import cv2
import face_recognition 
import os
import numpy as np 


rostos_codificados = []
nomes_rostos = []


for arquivo in os.listdir("rostos"):
  if arquivo.endswith(".npy"):
    nome = os.path.splitext(arquivo)[0]
    encoding = np.load(f"rostos/{arquivo}")
    rostos_codificados.append(encoding)
    nomes_rostos.append(nome)
    
    
print(f"{len(rostos_codificados)} rosto(s) carregado(s). Iniciando reconhecimento...")    

webcam = cv2.VideoCapture(0)

while True:
  ret, frame = webcam.read()
  if not ret:
    break
  
  pequena = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
  rgb = cv2.cvtColor(pequena, cv2.COLOR_BGR2RGB)
  
  localizacoes = face_recognition.face_locations(rgb)
  codificacoes = face_recognition.face_encodings(rgb,localizacoes)
  
  for(top, right, bottom, left), face_encoding in zip(localizacoes, codificacoes):
    resultados= face_recognition.compare_faces(rostos_codificados, face_encoding)
    distancias= face_recognition.face_distance(rostos_codificados, face_encoding)
    
    
    nome = "Desconhecido"
    if True in resultados:
      indice = np.argmin(distancias)
      nome = nomes_rostos[indice]
      
      
    top *= 4    
    right *= 4    
    bottom *= 4    
    left *= 4    
    
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
  
  
  cv2.imshow("Face in webcam", frame)
  
  if cv2.waitKey(5) ==27:
    break
  
  
webcam.release()
cv2.destroyAllWindows()  
  
  