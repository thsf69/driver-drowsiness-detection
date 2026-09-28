from config import CONSECUTIVE_FRAMES

class AlertManager:
    def __init__(self):
        self.consecutive_drowsy_frames = 0
        self.is_alerting = False

    def update(self, is_drowsy: bool):
        """
        Update the temporal logic based on the current frame's prediction.
        """
        if is_drowsy:
            self.consecutive_drowsy_frames += 1
            if self.consecutive_drowsy_frames >= CONSECUTIVE_FRAMES:
                self.is_alerting = True
        else:
            self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 2) # Smooth decay
            if self.consecutive_drowsy_frames == 0:
                self.is_alerting = False

        return self.is_alerting
        
    def get_status(self):
        return "ALERT" if self.is_alerting else "NORMAL"
