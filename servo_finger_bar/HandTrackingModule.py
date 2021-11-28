import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import time

class handDetector(): 
    def __init__(self,mode = False,maxHands= 2, detectionCon = 0.5,trackCon = 0.5): #fonksiynun sabit değerlerini giriyoruz
        # diğer fonksiyonlarda da kullanabilmek için değerlerimizi self şeklinde kaydediyoruz
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #bu kod parçasında, elleri algılamak için mp.solutions.hands'den "hands" adlı bir nesne bildiriyoruz 
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon) # this class only uses RGB img hands nesnesine sabit değerleri giriyoruz
        self.mpDraw = mp.solutions.drawing_utils # Ve kilit noktaları çizmek için mpDraw'ı kullanacağız .

    def findHands(self,img,draw = True): # elleri buldurmak için bir fonk oluşturuyoruz
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # hands nesnesine sadece rgb foto verilir o yuzden dönüştürme işlemi yapıyoz
        self.results = self.hands.process(imgRGB) # self.results yapınca her yerde her methodda kullanabiliyoz
        # print(results.multi_hand_landmarks) # elimizi koyduğumuzda bir sürü değer verir
        if self.results.multi_hand_landmarks: # eğer değerler varsa yani el bulunmuşsa img de
            for handLms in self.results.multi_hand_landmarks: # elin landmark larını alırız
               if draw: # eğer çizme işlemi trueysa 
                self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS) # aldığımız landmarklardan bgr foto uzerinde noktalardan oluşan bir görüntü elde ederiz, 3. methodla bu çizgileri birleştiririz 
        return img # fotonun son halini döndürdük
    def findPosition(self,img,handNo=0,draw=True): # landmarkların konumlarını x y bulmak için bir fonk oluşturuyoruz
                lmList = [] # boş bir liste oluşturuyoruz
                if self.results.multi_hand_landmarks: #eğer el veya eller varsa
                    myHand = self.results.multi_hand_landmarks[handNo] # elin konum bilgilerini alıyoruz x y z
                    for id,lm in enumerate(myHand.landmark): # eldeki noktaların ıd lerini ve lmlerini alıyoruz
                        # print(id,lm)
                        h,w,c = img.shape # image shapei bize uzunluk ve kalınlık ve channel
                        cx,cy = int(lm.x*w), int(lm.y*h) # değerleri piksel cinsine ceviriyoruz cevirmezsek 0.4547845 gibi verir
                        # print(id,cx,cy)
                        lmList.append([id,cx,cy]) # nokta ıdsi ve konumları piksel olarak
                        if draw:
                            cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED) # içi dolu çemberler çiziyoruz moktalara
                        # if id ==8:
                        #     cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        # lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,('FPS:'+str(int(fps))),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) # fps i buraya text olarak yazdırdık

        cv2.imshow('Image',img)
        if cv2.waitKey(20) & 0xFF ==27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

