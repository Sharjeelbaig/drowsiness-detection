import cv2
import dlib
import time
import numpy as np
import os
import sys


os.system("rm -r photos/*")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))
from config import EAR_THRESHOLD, MAR_THRESHOLD, PERCLOS_THRESHOLD, ALERT_INTERVAL
from utils import calculate_ear, calculate_mar, log_drowsiness_detected, face_recognize, draw_frame_rectanguler

def setup_tracker_and_models():
    return dlib.get_frontal_face_detector(), dlib.shape_predictor("../data/model.dat"), dlib.correlation_tracker()

def process_frame(frame, tracker, face_detector, dlib_facelandmark):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the original frame for overlay
    thumbnail_size = (250, 150)  # Adjust size as needed
    resized_original = cv2.resize(frame, thumbnail_size)

    # Ensure gray_frame is in BGR format for overlay
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Create a circular mask
    mask = np.zeros((thumbnail_size[1], thumbnail_size[0]), dtype=np.uint8)
    center = (thumbnail_size[0] // 2, thumbnail_size[1] // 2)
    radius = min(thumbnail_size) // 2
    # cv2.circle(mask, center, radius, (255), thickness=-1)
    # cv2.ellipse(mask, center, (radius, radius), 0, 0, 360, (255), thickness=-1)
    cv2.rectangle(mask, (0, 0), (thumbnail_size[0], thumbnail_size[1]), (255), thickness=-1)

    # Apply the circular mask to the resized original frame
    masked_resized = cv2.bitwise_and(resized_original, resized_original, mask=mask)

    # Calculate the top-right position for overlay
    x_offset = gray_frame_bgr.shape[1] - thumbnail_size[0] - 10  # 10-pixel margin
    y_offset = 10  # Top margin

    # Overlay the masked frame onto the top-right corner of the gray frame
    gray_frame_bgr[y_offset:y_offset + thumbnail_size[1], x_offset:x_offset + thumbnail_size[0]] = masked_resized

    if tracker.get_position().width() > 0:
        tracker.update(gray_frame)
        pos = tracker.get_position()
        face = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        landmarks = dlib_facelandmark(gray_frame, face)

        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]

        return gray_frame_bgr, pos, left_eye, right_eye, mouth
    else:
        faces = face_detector(gray_frame)
        if faces:
            tracker.start_track(gray_frame, faces[0])
        return gray_frame_bgr, None, None, None, None


def display_debug_info(frame, smoothed_ear, mar, perclos, cycle):
    cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Cycle: {cycle}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def handle_drowsiness_alert(smoothed_ear, mar, perclos, drowsiness_counter, last_alert_time, ):
    
    if smoothed_ear < EAR_THRESHOLD or mar > MAR_THRESHOLD or perclos > PERCLOS_THRESHOLD:
        drowsiness_counter += 1
        if drowsiness_counter > 5 and time.time() - last_alert_time > ALERT_INTERVAL:
            log_drowsiness_detected(f"Drowsiness detected at {time.ctime()}")
            last_alert_time = time.time()
            drowsiness_counter = 0
            is_drowsy = True
    else:
        drowsiness_counter = 0        
    return drowsiness_counter, last_alert_time, 

def main():
    recognized = False
    face_detector, dlib_facelandmark, tracker = setup_tracker_and_models()
    cap = cv2.VideoCapture(0)

    smooth_ear = []
    CLOSED_EYE_FRAMES, TOTAL_FRAMES, cycle, recognition_cycle, drowsiness_counter = 0, 0, 0, 0, 0
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame, pos, left_eye, right_eye, mouth = process_frame(frame, tracker, face_detector, dlib_facelandmark)
        
        if pos and left_eye and right_eye and mouth:
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            mar = calculate_mar(mouth)

            smooth_ear.append(avg_ear)
            if len(smooth_ear) > 10:
                smooth_ear.pop(0)
            smoothed_ear = np.mean(smooth_ear)

            draw_frame_rectanguler(gray_frame, pos, f"Target {'Recognized' if (recognized and cycle > 100) else "Verifying" if cycle else 'Unrecognized'}")

            TOTAL_FRAMES += 1
            if smoothed_ear < EAR_THRESHOLD:
                CLOSED_EYE_FRAMES += 1

            perclos = (CLOSED_EYE_FRAMES / TOTAL_FRAMES) * 100
            display_debug_info(gray_frame, smoothed_ear, mar, perclos, cycle)

            
            
            drowsiness_counter, last_alert_time = handle_drowsiness_alert(smoothed_ear, mar, perclos, drowsiness_counter, last_alert_time)

            if drowsiness_counter > 5:
                cv2.putText(gray_frame, "DROWSINESS ALERT!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drowsiness_counter = 0
            


        cv2.imwrite(f'../photos/{TOTAL_FRAMES}.png', gray_frame)

        if recognition_cycle == 100:
            try:
                if face_recognize(gray_frame):
                    # text on bottom left corner
                    recognized = True
                    print('Face recognized')
                else:
                    recognized = False
                    print('Face not recognized')
                    continue
            except Exception as e:
                print(f"Error in face recognition: {e}")
                continue

        recognition_cycle = (recognition_cycle + 1) % 101
        cycle += 1

        cv2.imshow("Debug View", gray_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
