import cv2

cap = cv2.VideoCapture(0) 
eyesdetector = cv2.CascadeClassifier("third-party/frontalEyes35x16.xml")

while True:
    
    overlay=cv2.imread("glasses.png",cv2.IMREAD_UNCHANGED)
    

    ret,frame = cap.read() 
            
    if ret ==False:
        continue
    
    eyes=eyesdetector.detectMultiScale(frame,1.1,5)
    for eye in eyes:

        x,y,w,h=eye
        overlay=cv2.resize(overlay,(w,h))

        


        for i in range(overlay.shape[0]): 
            for j in range(overlay.shape[1]): 
                if(overlay[i,j,3]>0):  
                    frame[y+i,x+j,:]=overlay[i,j,:-1]  
            
    
    cv2.imshow("sd",frame)



    key_pressed = cv2.waitKey(1) & 0xFF 

   


    if key_pressed==ord('q'):
        print(key_pressed)
        print(ord('q'))
        break
    


cap.release()
cv2.destroyAllWindows()
