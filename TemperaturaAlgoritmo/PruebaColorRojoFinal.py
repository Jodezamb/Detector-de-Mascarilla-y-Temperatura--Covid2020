#importas las librerias
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#Limites  y rango del color rojo Prueba 2
LimiteinferiorRojo1 = np.array([0, 100, 20], np.uint8)
LimiteSuperiorRojo1 = np.array([1, 255, 255], np.uint8)

#Segunda rango de limte color rojo seguna codigo de colocre hsv
LimiteinferiorRojo2 = np.array([175, 100, 20], np.uint8)
LimiteSuperiorRojo2 = np.array([179, 255, 255], np.uint8)


while True: # se inicia la camara  permiendo leer la imagen a cada momento 
  ret,frame = cap.read()

  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #submascaras del color rojo 
    maskroja1 = cv2.inRange(frameHSV, LimiteinferiorRojo1, LimiteSuperiorRojo1)
    maskroja2 = cv2.inRange(frameHSV, LimiteinferiorRojo2, LimiteSuperiorRojo2)
    #mascara de color rojo con sus rangos
    maskroja=cv2.add(maskroja1,maskroja2)
    contornos,variable = cv2.findContours(maskroja, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #mostras contornos en el video 
    #primera prueba
    #cv2.drawContours(frame, contornos, -1, (0,255,0), 3)

    #seleccion de contornos 
    for c in contornos:
        area=cv2.contourArea(c)
        if area > 3000: 
          #Buscar el area central del objeto especificado para  mostrarla
          M = cv2.moments(c)
          if (M["m00"]==0): M["m00"]=1
          #valores de x,y 
          x = int(M["m10"]/M["m00"])
          y = int(M['m01']/M['m00'])
          # se dibuja el circulo
          cv2.circle(frame, (x,y), 7, (0,255,0), -1)
          #fuente del texto 
          font = cv2.FONT_HERSHEY_SIMPLEX
          #colancandolo en la imagen
          cv2.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.75,(0,255,0),1,cv2.LINE_AA)
          # suavizado de lineas
          newContourn=cv2.convexHull(c)
          cv2.drawContours(frame, [newContourn],0, (0,255,0), 3) # para dibujar solo ciertos contornos 
            
    cv2.imshow('frame', frame)
   # cv2.imshow('maskRed', maskazul)
    #dIBUJAR LOS CONTRONOS ENCONTRADOS\
    
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()