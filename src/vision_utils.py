import math

# Facial landmarks for eyes and mouth in MediaPipe Face Mesh
# Left eye: 33, 160, 158, 133, 153, 144
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# Right eye: 362, 385, 387, 263, 373, 380
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth (inner lips): 78, 81, 13, 311, 308, 402, 14, 178
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks):
    # Left Eye
    le_v1 = euclidean_distance(landmarks[160], landmarks[144])
    le_v2 = euclidean_distance(landmarks[158], landmarks[153])
    le_h = euclidean_distance(landmarks[33], landmarks[133])
    left_ear = (le_v1 + le_v2) / (2.0 * le_h)
    
    # Right Eye
    re_v1 = euclidean_distance(landmarks[385], landmarks[380])
    re_v2 = euclidean_distance(landmarks[387], landmarks[373])
    re_h = euclidean_distance(landmarks[362], landmarks[263])
    right_ear = (re_v1 + re_v2) / (2.0 * re_h)
    
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks):
    # Vertical distances
    v1 = euclidean_distance(landmarks[81], landmarks[178])
    v2 = euclidean_distance(landmarks[13], landmarks[14])
    v3 = euclidean_distance(landmarks[311], landmarks[402])
    
    # Horizontal distance
    h = euclidean_distance(landmarks[78], landmarks[308])
    
    return (v1 + v2 + v3) / (2.0 * h)
