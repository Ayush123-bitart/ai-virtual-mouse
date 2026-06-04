import cv2
import mediapipe as mp
import pyautogui
import math

# Webcam Start
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing Utility
mpDraw = mp.solutions.drawing_utils

# Screen Size
screen_width, screen_height = pyautogui.size()

# Smooth Cursor Variables
prev_x = 0
prev_y = 0
smoothening = 5

while True:

    # Read Camera Frame
    success, img = cap.read()

    if not success:
        break

    # Flip Image
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process Hand
    results = hands.process(imgRGB)

    # If Hand Detected
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            landmark_list = []

            # Get Landmark Positions
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmark_list.append((id, cx, cy))

            # Draw Hand Landmarks
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            if len(landmark_list) != 0:

                # --------------------------------
                # Index Finger = Cursor Movement
                # --------------------------------

                x_index = landmark_list[8][1]
                y_index = landmark_list[8][2]

                # Draw Circle on Index Finger
                cv2.circle(
                    img,
                    (x_index, y_index),
                    15,
                    (255, 0, 255),
                    cv2.FILLED
                )

                # Convert Camera Coordinates to Screen Coordinates
                mouse_x = screen_width / w * x_index
                mouse_y = screen_height / h * y_index

                # Smooth Cursor Movement
                curr_x = prev_x + (mouse_x - prev_x) / smoothening
                curr_y = prev_y + (mouse_y - prev_y) / smoothening

                # Move Mouse
                pyautogui.moveTo(curr_x, curr_y)

                prev_x = curr_x
                prev_y = curr_y

                # --------------------------------
                # Thumb + Middle Finger = Click
                # --------------------------------

                x_thumb = landmark_list[4][1]
                y_thumb = landmark_list[4][2]

                x_middle = landmark_list[12][1]
                y_middle = landmark_list[12][2]

                # Distance Between Thumb and Middle Finger
                distance = math.hypot(
                    x_middle - x_thumb,
                    y_middle - y_thumb
                )

                # Draw Line
                cv2.line(
                    img,
                    (x_thumb, y_thumb),
                    (x_middle, y_middle),
                    (0, 255, 0),
                    3
                )

                # Click Gesture
                if distance < 25:

                    pyautogui.click()

                    # Small Delay
                    pyautogui.sleep(0.2)

    # Show Window
    cv2.imshow("Virtual Mouse", img)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Camera
cap.release()
cv2.destroyAllWindows()