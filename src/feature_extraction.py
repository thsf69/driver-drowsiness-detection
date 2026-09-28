import pandas as pd
from config import ROLLING_WINDOW_SIZE

def extract_temporal_features(df):
    """
    Extract temporal features from raw EAR and MAR data.
    Adds rolling means and rolling standard deviations to capture sustained changes.
    """
    df = df.copy()
    
    if 'session_id' in df.columns:
        # Calculate rolling statistics grouped by session to prevent leakage between independent videos
        df['EAR_rolling_mean'] = df.groupby('session_id')['EAR'].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean())
        df['EAR_rolling_std'] = df.groupby('session_id')['EAR'].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std().fillna(0))
        
        df['MAR_rolling_mean'] = df.groupby('session_id')['MAR'].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean())
        df['MAR_rolling_std'] = df.groupby('session_id')['MAR'].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std().fillna(0))
    else:
        # Fallback for real-time inference on a single continuous stream
        df['EAR_rolling_mean'] = df['EAR'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
        df['EAR_rolling_std'] = df['EAR'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std().fillna(0)
        
        df['MAR_rolling_mean'] = df['MAR'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
        df['MAR_rolling_std'] = df['MAR'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std().fillna(0)
        
    return df
