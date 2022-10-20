import cv2
import mediapipe as mp
import time     # to check frame rate

cap = cv2.VideoCapture(0)       # initializing video input from default webcam


mpHands= mp.solutions.hands
hands= mpHands.Hands()

mpDraw= mp.solutions.drawing_utils      #draws points around the hand landmarks

pTime= 0
cTime= 0        # initializing previous and curent times for determininng frame rate

while True:
    success, img= cap.read()        # this gives us the frame
    # sending the rgb image to the object
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # the hands object created earlier takes only rgb images
    results= hands.process(imgRGB)      #processing the image
    # print(results.multi_hand_landmarks)     #shows realtime hand values when hand is placed before the camera, removing the hand displays None

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:        # handlms is the short for hand landmarks
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    cTime= time.time()
    fps= 1/(cTime-pTime)
    pTime= cTime

    '''
    Display the fps on the screen:
    (10,70)- the location of fps text on screen
    font is FONT_HERSHLEY_PLAIN
    3 is the scale
    color is purple the code for which is (255,0,255)
    2 is the thickness'''

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)   

    cv2.imshow("Image", img)
    cv2.waitKey(1)
