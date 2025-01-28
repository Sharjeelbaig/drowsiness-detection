import numpy as np
from scipy.spatial import distance
import face_recognition
import cv2

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def face_recognize(frame):
    temp_dir = '../temp'
    os.makedirs(temp_dir, exist_ok=True)
    # convert frame to jpg
    cv2.imwrite(f'{temp_dir}/frame.jpg', frame)
    # load image file
    image = face_recognition.load_image_file(f'{temp_dir}/frame.jpg')
    image_encoding = face_recognition.face_encodings(image)[0]
    # load known image file
    known_image_dir = '../database/known.png'
    known_image = face_recognition.load_image_file(known_image_dir)
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    # compare faces
    result = face_recognition.compare_faces([known_image_encoding], image_encoding)
    if result[0]:
        print('Face recognized')
        return True
    else:
        print('Face not recognized')
        return False
    

def draw_frame_rectanguler(frame, pos, text):
    # Define target rectangle parameters
    center_x = int((pos.left() + pos.right()) / 2)
    center_y = int((pos.top() + pos.bottom()) / 2)
    width = int(pos.right() - pos.left())
    height = int(pos.bottom() - pos.top())

    # Draw horizontal and vertical lines intersecting at the center
    cv2.line(frame, (center_x - width // 4, center_y), (center_x + width // 4, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - height // 4), (center_x, center_y + height // 4), (0, 255, 0), 2)

    # Draw outer box corners (target-style bounding box)
    corner_length = 20
    # Top-left corner
    cv2.line(frame, (int(pos.left()), int(pos.top())), (int(pos.left() + corner_length), int(pos.top())), (0, 255, 0), 2)
    cv2.line(frame, (int(pos.left()), int(pos.top())), (int(pos.left()), int(pos.top() + corner_length)), (0, 255, 0), 2)
    # Top-right corner
    cv2.line(frame, (int(pos.right()), int(pos.top())), (int(pos.right() - corner_length), int(pos.top())), (0, 255, 0), 2)
    cv2.line(frame, (int(pos.right()), int(pos.top())), (int(pos.right()), int(pos.top() + corner_length)), (0, 255, 0), 2)
    # Bottom-left corner
    cv2.line(frame, (int(pos.left()), int(pos.bottom())), (int(pos.left() + corner_length), int(pos.bottom())), (0, 255, 0), 2)
    cv2.line(frame, (int(pos.left()), int(pos.bottom())), (int(pos.left()), int(pos.bottom() - corner_length)), (0, 255, 0), 2)
    # Bottom-right corner
    cv2.line(frame, (int(pos.right()), int(pos.bottom())), (int(pos.right() - corner_length), int(pos.bottom())), (0, 255, 0), 2)
    cv2.line(frame, (int(pos.right()), int(pos.bottom())), (int(pos.right()), int(pos.bottom() - corner_length)), (0, 255, 0), 2)

    # Draw text below the crosshair
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = center_x - text_size[0] // 2  # Center the text
    text_y = center_y + height // 2 + 20  # Position below the crosshair
    cv2.putText(frame, text, (text_x, text_y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)






import os

# Save logs to a file
def log_drowsiness_detected(message):
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'drowsiness_log.txt'), 'a') as log_file:
        log_file.write(message + "\n")
