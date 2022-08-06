import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0

while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for  handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)
