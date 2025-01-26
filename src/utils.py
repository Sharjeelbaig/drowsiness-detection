import numpy as np
from scipy.spatial import distance

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

import os

# Save logs to a file
def log_drowsiness_detected(message):
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'drowsiness_log.txt'), 'a') as log_file:
        log_file.write(message + "\n")
