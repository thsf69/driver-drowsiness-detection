from src.vision_utils import calculate_ear, calculate_mar
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

def test_euclidean_distance_and_ear():
    # Mock landmarks for a perfect square eye (width 2, height 2)
    # 33: (0,1), 133: (2,1)
    # 160: (0.5, 2), 158: (1.5, 2)
    # 153: (1.5, 0), 144: (0.5, 0)
    
    # Just checking the functions run and return expected math
    landmarks = [Point(0,0)] * 468
    
    # Let's just mock the needed ones for left eye to see if it doesn't crash
    landmarks[33] = Point(0, 1)
    landmarks[133] = Point(2, 1)
    landmarks[160] = Point(0.5, 2)
    landmarks[144] = Point(0.5, 0)
    landmarks[158] = Point(1.5, 2)
    landmarks[153] = Point(1.5, 0)
    
    # Right eye mock
    landmarks[362] = Point(0, 1)
    landmarks[263] = Point(2, 1)
    landmarks[385] = Point(0.5, 2)
    landmarks[380] = Point(0.5, 0)
    landmarks[387] = Point(1.5, 2)
    landmarks[373] = Point(1.5, 0)
    
    ear = calculate_ear(landmarks)
    assert ear > 0
    
    # Test MAR
    landmarks[81] = Point(1, 2)
    landmarks[178] = Point(1, 0)
    landmarks[13] = Point(1.5, 2)
    landmarks[14] = Point(1.5, 0)
    landmarks[311] = Point(2, 2)
    landmarks[402] = Point(2, 0)
    
    landmarks[78] = Point(0, 1)
    landmarks[308] = Point(3, 1)
    
    mar = calculate_mar(landmarks)
    assert mar > 0
