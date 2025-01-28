import cv2
import dlib
import time
import numpy as np
import os
import sys


os.system("rm -r photos/*")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))

from config import EAR_THRESHOLD, MAR_THRESHOLD, PERCLOS_THRESHOLD, ALERT_INTERVAL
from utils import calculate_ear, calculate_mar, log_drowsiness_detected, face_recognize,draw_frame_rectanguler

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("../data/model.dat")

CLOSED_EYE_FRAMES = 0
TOTAL_FRAMES = 0
last_alert_time = 0
drowsiness_counter = 0
smooth_ear = []

# Initialize the correlation tracker
tracker = dlib.correlation_tracker()

cycle = 0
recognition_cycle = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # main_frame = gray.copy()
    main_frame = gray.copy()


    if tracker.get_position().width() > 0:
        # Track the face using the tracker
        tracker.update(main_frame)
        pos = tracker.get_position()
        face = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        landmarks = dlib_facelandmark(main_frame, face)

        # Get coordinates for eyes
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Get coordinates for mouth
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]

        # Calculate EAR and MAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        mar = calculate_mar(mouth)

        # Smooth EAR values for better accuracy
        smooth_ear.append(avg_ear)
        if len(smooth_ear) > 10:  # Keep a rolling window of the last 10 frames
            smooth_ear.pop(0)
        smoothed_ear = np.mean(smooth_ear)

        # rectanguled_frame = cv2.rectangle(main_frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 2)
        draw_frame_rectanguler(main_frame, pos)

        # Count closed-eye frames for PERCLOS
        TOTAL_FRAMES += 1
        if smoothed_ear < EAR_THRESHOLD:
            CLOSED_EYE_FRAMES += 1

        # Calculate PERCLOS
        perclos = (CLOSED_EYE_FRAMES / TOTAL_FRAMES) * 100

        # Display EAR, MAR, and PERCLOS values
        cv2.putText(main_frame, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(main_frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(main_frame, f"PERCLOS: {perclos:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(main_frame, f"cycle: {cycle}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Trigger alert if drowsiness is detected
        if smoothed_ear < EAR_THRESHOLD or mar > MAR_THRESHOLD or perclos > PERCLOS_THRESHOLD:
            drowsiness_counter += 1
            cv2.putText(main_frame, "DROWSY", (main_frame.shape[1] - 350, main_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if drowsiness_counter > 5:  # Trigger alert only after 5 consecutive detections
                if time.time() - last_alert_time > ALERT_INTERVAL:
                    log_drowsiness_detected(f"Drowsiness detected at {time.ctime()}")
                    last_alert_time = time.time()
                    drowsiness_counter = 0  # Reset the counter after alert
        else:
            drowsiness_counter = 0  # Reset if drowsiness is not detected

    else:
        # Detect a new face if the tracker is lost
        faces = face_detector(main_frame)
        if faces:
            face = faces[0]
            tracker.start_track(main_frame, face)

     # write ppng to ../photos/1.png, ../photos/2.png, ...
    cv2.imwrite(f'../photos/{TOTAL_FRAMES}.png', main_frame)

    
    
    # Show the frame as debug view
    cv2.imshow("debug view", main_frame)


    cycle = cycle+1
    recognition_cycle = recognition_cycle+1

    if recognition_cycle == 100:
        # recognize face
        try:
            if face_recognize(main_frame):
                print('Face recognized')
            else:
                print('Face not recognized')
                continue
        except Exception as e:
            print(f"Error in face recognition: {e}")
            continue
    
    if recognition_cycle > 101:
        recognition_cycle = 0
    
    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
