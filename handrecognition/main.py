import cv2
import os
import mediapipe as mp
import uuid


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#run the below line, when the code is running for first time, later it can be commented out.

#os.mkdir('Output Images')

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # passing the feed to mediapipe to find the hand
        result = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(results)

        # embedding the detected points of the hands on to the feed
        if result.multi_hand_landmarks:
            for num, hand in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=2, circle_radius=2),
                                          )
                # Saving the results, this line can be commented if you don't want to save results
                cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
