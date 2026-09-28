import os
import numpy as np
import pandas as pd
from config import PROCESSED_DATA_PATH

def generate_alert_sequence(num_frames):
    # Base EAR for alert person is around 0.28 to 0.35
    ear = np.random.normal(0.30, 0.02, num_frames)
    # Base MAR for alert person is around 0.3 to 0.4
    mar = np.random.normal(0.35, 0.05, num_frames)
    
    # Add random short blinks (EAR drops significantly)
    num_blinks = int(num_frames / 30)  # Roughly one blink every 30 frames
    for _ in range(num_blinks):
        blink_start = np.random.randint(0, num_frames - 5)
        blink_duration = np.random.randint(1, 4)
        ear[blink_start:blink_start+blink_duration] = np.random.normal(0.15, 0.02, blink_duration)
    
    return ear, mar, np.zeros(num_frames)  # Label 0 for Alert

def generate_drowsy_sequence(num_frames):
    # Base EAR for drowsy person is lower, around 0.20 to 0.25 (drooping eyelids)
    ear = np.random.normal(0.22, 0.03, num_frames)
    # Base MAR might be higher due to yawning
    mar = np.random.normal(0.50, 0.1, num_frames)
    
    # Add long blinks / microsleeps (EAR drops significantly for longer periods)
    num_blinks = int(num_frames / 40)
    for _ in range(num_blinks):
        blink_start = np.random.randint(0, num_frames - 15)
        blink_duration = np.random.randint(5, 15) # Longer closure
        ear[blink_start:blink_start+blink_duration] = np.random.normal(0.12, 0.02, blink_duration)
    
    # Add some yawning sequences
    num_yawns = int(num_frames / 100)
    for _ in range(num_yawns):
        yawn_start = np.random.randint(0, num_frames - 30)
        yawn_duration = np.random.randint(15, 30)
        mar[yawn_start:yawn_start+yawn_duration] = np.random.normal(0.75, 0.05, yawn_duration)
        
    return ear, mar, np.ones(num_frames) # Label 1 for Drowsy

def main():
    print("Generating synthetic dataset for testing/training pipeline...")
    np.random.seed(42)
    
    data = []
    # Generate multiple sessions to create variance
    for session in range(50):
        if session % 2 == 0:
            e, m, l = generate_alert_sequence(200) # 200 frames per sequence
        else:
            e, m, l = generate_drowsy_sequence(200)
            
        df = pd.DataFrame({'EAR': e, 'MAR': m, 'label': l})
        df['session_id'] = session  # Crucial for preventing data leakage
        data.append(df)
        
    final_df = pd.concat(data, ignore_index=True)
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Dataset generated with {len(final_df)} frames and saved to {PROCESSED_DATA_PATH}.")

if __name__ == "__main__":
    main()
