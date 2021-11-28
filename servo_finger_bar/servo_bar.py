import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import pyfirmata
import time


cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.handDetector(detectionCon=0.7,maxHands= 1) # en fazla 1 el bul

board=pyfirmata.Arduino('COM6')

iter8 = pyfirmata.util.Iterator(board)
iter8.start()

pin9 = board.get_pin('d:3:s')

vol = 0
volBar = 0
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        # print(lmList[4],lmList[8])
        x1,y1 = lmList[4][1] , lmList[4][2]
        x2,y2 = lmList[8][1] , lmList[8][2]
        cx , cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1) # length alıp aralıklara cevirttik
        # print(length)
        # Hand Range 15 145
        # Volume RAnge - 96 0
        vl =  np.interp(length,[15,143],[130,8])
        volBar = np.interp(length,[15,143],[400,150]) # barın uzunluğuna oaranlattık
        volPer = np.interp(length,[15,143],[0,100]) # yüzdeye oranlattık
        pin9.write(vl)
        
        # print(int(length),vol)
        if length<15:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)

        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3) # bir dikdörtgen çizdirdim
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,0,255),cv2.FILLED) # içi dolu başka dikd çizdirdim
        cv2.putText(img,f'%{int(volPer)}', (40,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3) # percentage i yazdım

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
    cv2.imshow('Img',img)
    if  cv2.waitKey(20) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

 