import cv2 as cv
import HandTrackingModule as htm
import streamlit as st

st.title('Finger Counter')
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])
capture = cv.VideoCapture(0)

detector = htm.HandDetector(maxHands=1)

tipIds = [4, 8, 12, 16, 20]

while run:
    success, img = capture.read()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[4][1] < lmList[20][1]: # Right hand

            # For thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        else: # Left hand

            # For thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        cv.putText(img, f'Fingers: {fingers.count(1)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    FRAME_WINDOW.image(img)