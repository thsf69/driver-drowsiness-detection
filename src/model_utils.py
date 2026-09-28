import pickle
import os
from config import MODEL_PATH, SCALER_PATH, MODEL_DIR

def save_model(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Please train the model first.")
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler
