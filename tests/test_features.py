import numpy as np
import pandas as pd
from src.feature_extraction import extract_temporal_features

def test_extract_temporal_features():
    df = pd.DataFrame({
        'session_id': [0, 0, 0, 1, 1],
        'EAR': [0.3, 0.3, 0.2, 0.2, 0.3],
        'MAR': [0.4, 0.4, 0.5, 0.5, 0.4]
    })
    
    processed = extract_temporal_features(df)
    
    assert 'EAR_rolling_mean' in processed.columns
    assert 'EAR_rolling_std' in processed.columns
    assert 'MAR_rolling_mean' in processed.columns
    assert 'MAR_rolling_std' in processed.columns
    
    assert len(processed) == 5
    assert not processed['EAR_rolling_mean'].isna().all()
