import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from config import CAMERA_INDEX, ROLLING_WINDOW_SIZE
from vision_utils import calculate_ear, calculate_mar
from model_utils import load_model
from alert_manager import AlertManager

def main():
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}. Please check your camera connection.")
        return

    alert_manager = AlertManager()
    
    # History for rolling features
    ear_history = deque(maxlen=ROLLING_WINDOW_SIZE)
    mar_history = deque(maxlen=ROLLING_WINDOW_SIZE)
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for natural viewing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                ear = calculate_ear(landmarks)
                mar = calculate_mar(landmarks)
                
                ear_history.append(ear)
                mar_history.append(mar)
                
                # Check if we have enough history to make a prediction
                if len(ear_history) == ROLLING_WINDOW_SIZE:
                    ear_mean = np.mean(ear_history)
                    ear_std = np.std(ear_history)
                    mar_mean = np.mean(mar_history)
                    mar_std = np.std(mar_history)
                    
                    # Create feature vector matching training data
                    features = pd.DataFrame([{
                        'EAR': ear,
                        'MAR': mar,
                        'EAR_rolling_mean': ear_mean,
                        'EAR_rolling_std': ear_std,
                        'MAR_rolling_mean': mar_mean,
                        'MAR_rolling_std': mar_std
                    }])
                    
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0]
                    confidence = max(prob) * 100
                    
                    is_drowsy = bool(prediction == 1)
                    
                    # Update temporal logic
                    is_alerting = alert_manager.update(is_drowsy)
                    status = alert_manager.get_status()
                    
                    # Drawing info on frame
                    color = (0, 0, 255) if is_alerting else (0, 255, 0)
                    
                    cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Drowsy Frames: {alert_manager.consecutive_drowsy_frames}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Initializing...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Driver Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
