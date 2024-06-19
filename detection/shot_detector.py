# shot logic:
# - keep track of latest player within certain distance of ball (might need another model to track ball)
# - if ball is a certain threshold above rim, it is a shot


class ShotDetector:
    def __init__(self, rim_y: int, ball_y: int, threshold: int = 100):
        self.threshold = threshold
        self.difference = ball_y - rim_y

    def detect_shot(self, rim_y: int, ball_y: int):
        if self.difference >= self.threshold:
            return True
        return False
