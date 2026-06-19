import cv2
import mediapipe as mp
import math
import pyautogui
import os
import time

# Screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

clicking = False
right_clicking = False
screenshotting = False

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            pinky = hand_landmarks.landmark[20]

            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)
            middle_x = int(middle_finger.x * frame_width)
            middle_y = int(middle_finger.y * frame_height)
            pinky_x = int(pinky.x * frame_width)
            pinky_y = int(pinky.y * frame_height)

            # Draw dots
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            # Move mouse cursor
            screen_x = screen_width * index_finger.x
            screen_y = screen_height * index_finger.y

            pyautogui.moveTo(screen_x, screen_y)

            # Distance for click
            distance = math.hypot(
                thumb_x - index_x,
                thumb_y - index_y
            )

            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # Click gesture
            if distance < 40:
                if not clicking:
                    print("CLICK!")
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False

            middle_distance = math.hypot(
                thumb_x - middle_x,
                thumb_y - middle_y
            )

            if middle_distance < 40:
                if not right_clicking:
                    print("RIGHT CLICK!")
                    pyautogui.rightClick()
                    right_clicking = True
            else:
                right_clicking = False

            pinky_distance = math.hypot(
                thumb_x - pinky_x,
                thumb_y - pinky_y
            )

            if pinky_distance < 40:
                if not screenshotting:

                    filename = f"screenshots/screenshot_{int(time.time())}.png"

                    screenshot = pyautogui.screenshot()
                    screenshot.save(filename)

                print("SCREENSHOT SAVED!")

                screenshotting = True

            else:
                screenshotting = False

    cv2.imshow("AI Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()