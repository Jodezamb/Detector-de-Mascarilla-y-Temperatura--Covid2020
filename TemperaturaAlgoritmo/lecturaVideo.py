import cv2
import numpy as np

captura = cv2.VideoCapture(0) #se activa la camara de la pc

#Para guardar el video 
# salida=cv2.VideoWriter('videosalida',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))

rebbajo1=np.array([0,100,20],np.uint8) # rango de colores en hsv
rebbaltoo1=np.array([0,255,255],np.uint8) # rango de colores en hsv

rebbajo2=np.array([175,100,20],np.uint8) # rango de colores en hsv
rebbaltoo2=np.array([179,255,255],np.uint8) # rango de colores en hsv


while True: # se inicia la camara  permiendo leer la imagen a cada momento 
  ret, frame = captura.read() # comienzas  grabar en tiempo real para tomar el video  par caprutras o leer la imagen a cada momento  el ret cuando ya tenemos la imagen leida  y false cuando no ha inicializado la imagen
  if ret == True:  # cuuando ya 3tenemos la imagen leida  por eso le ponemos true para siempre obtener imagen 
      frameHSV=cv2.cvtColor(frame.cv2.Color_BGR2HSV) #TRANFORMAR DEL espacio BGR al espacio HSV 
      maskRed1=cv2.inRange(frameHSV,rebbajo1,rebbaltoo1)  
      maskRed2=cv2.inRange(frameHSV,rebbajo2,rebbaltoo2)  
      maskRed=cv2.add(maskRed1,maskRed2)

      #guardar al salidad 
      #salida.write(frame)
      
      #visualizacionn
      cv2.inshow('maskRed',maskRed) # inshow se utiliza para visaulizar la imagen 
      cv2.imshow('video', frame) # % lo que vamos a mostrar
      if cv2.waitKey(1) & 0xFF == ord('s'): # para cerrar la pestana y no quede abierta, y se escoge al laetra para cerrar la pestana  es para icualizar la imagen por un tiempo determinado 
        break # romperemos el lazo 
  
captura.release() # se finaliza la caotura
cv2.destroyAllWindows() # cerrar las ventanas que quedaron abiertas