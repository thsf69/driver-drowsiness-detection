import os
import pytest
from src.alert_manager import AlertManager
from src.config import CONSECUTIVE_FRAMES

def test_alert_manager_temporal_logic():
    manager = AlertManager()
    
    assert manager.get_status() == "NORMAL"
    
    # Send some normal frames
    manager.update(False)
    assert manager.get_status() == "NORMAL"
    
    # Send drowsy frames just below threshold
    for _ in range(CONSECUTIVE_FRAMES - 1):
        manager.update(True)
    assert manager.get_status() == "NORMAL"
    
    # One more should trigger alert
    manager.update(True)
    assert manager.get_status() == "ALERT"
    
    # Send normal frames to recover
    manager.update(False)
    manager.update(False)
    # The recovery takes a few frames to wind down
    # With decay = 2, from 20 -> 18 -> 16...
    assert manager.get_status() == "ALERT"
